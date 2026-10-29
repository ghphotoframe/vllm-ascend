# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Packed Ling conv weights use the real initial-load and RL reload drivers."""

from itertools import permutations
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._python_dispatch import TorchDispatchMode
from vllm.model_executor.layers.attention import is_deferred_attention_layer
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.bailing_moe_v3 import BailingMoeV3KimiDeltaAttention

from vllm_ascend.ops.bailing_moe_v3_kda import (
    _PACKED_CONV_WEIGHT_NAME,
    AscendBailingMoeV3KimiDeltaAttention,
)

_CONVS = ("q_conv1d", "k_conv1d", "v_conv1d")


class _PostLoadConv(QuantizeMethodBase):
    """A child postload transform makes packing-before-children observable."""

    def __init__(self, name, events):
        self.name = name
        self.events = events

    def create_weights(self, layer, *args, **kwargs):
        raise NotImplementedError

    def apply(self, layer, *args, **kwargs):
        raise NotImplementedError

    def process_weights_after_loading(self, layer):
        self.events.append(self.name)
        layer.weight.data.add_(0.25)


def _new_attention(dtype=torch.bfloat16, device="cpu"):
    config = SimpleNamespace(dtype=dtype, quantization=None, word_embeddings_untied_by_checkpoint=False)
    events = []

    def init(instance, *_args, **_kwargs):
        nn.Module.__init__(instance)
        instance.num_speculative_tokens = 0
        instance.conv_size = 4
        instance.projection_size_per_partition = 6
        instance.model_config = config
        for index, name in enumerate(_CONVS):
            child = nn.Module()
            source = torch.arange(24, dtype=torch.float32, device=device).reshape(6, 1, 4) / 16 + index
            child.weight = nn.Parameter(source, requires_grad=False)
            child.weight.weight_loader = default_weight_loader
            child.quant_method = _PostLoadConv(name, events)
            setattr(instance, name, child)
        for name in ("A_log", "dt_bias"):
            weight = nn.Parameter(torch.zeros(6, dtype=torch.float32, device=device), requires_grad=False)
            weight.weight_loader = default_weight_loader
            instance.register_parameter(name, weight)

    with (
        patch.object(BailingMoeV3KimiDeltaAttention, "__init__", init),
        patch("vllm_ascend.ops.bailing_moe_v3_kda.is_conv_state_dim_first", return_value=False),
    ):
        attention = AscendBailingMoeV3KimiDeltaAttention(None)
    model = nn.Sequential(attention)
    record_metadata_for_reloading(model)
    return model, attention, config, events


def _expected(attention, dtype):
    return torch.cat([getattr(attention, name).weight[:, 0, :] for name in _CONVS], dim=0).T.to(dtype).contiguous()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@torch.no_grad()
def test_initial_load_defers_packing_until_all_child_postloads(dtype):
    model, attention, config, events = _new_attention(dtype)
    packed = attention.get_buffer(_PACKED_CONV_WEIGHT_NAME)
    original_ptr = packed.data_ptr()
    assert is_deferred_attention_layer(attention)
    with pytest.raises(RuntimeError, match="prepared after weight loading"):
        attention._conv_weights_t(dtype)

    process_weights_after_loading(model, config, torch.device("cpu"))

    assert events == list(_CONVS)
    assert attention._conv_weights_t(dtype) is packed
    assert packed.data_ptr() == original_ptr
    assert packed.dtype == dtype and packed.is_contiguous()
    torch.testing.assert_close(packed, _expected(attention, dtype), rtol=0, atol=0)


@pytest.mark.parametrize("order", list(permutations(_CONVS)))
@torch.no_grad()
def test_two_rl_updates_preserve_packed_storage_and_latest_three_weights(order):
    model, attention, config, events = _new_attention()
    process_weights_after_loading(model, config, torch.device("cpu"))
    captured = attention._conv_weights_t(config.dtype)
    original_parameters = dict(attention.named_parameters())
    original_ptrs = {name: tensor.data_ptr() for name, tensor in original_parameters.items()}

    for generation in (1, 2):
        events.clear()
        before = captured.clone()
        initialize_layerwise_reload(model)
        assert attention.get_buffer(_PACKED_CONV_WEIGHT_NAME).is_meta
        with pytest.raises(RuntimeError, match="prepared after weight loading"):
            attention._conv_weights_t(config.dtype)
        # Include parent-owned checkpoint parameters before the children;
        # their loaders must not trigger an early pack or replace graph storage.
        for name in ("A_log", "dt_bias"):
            param = getattr(attention, name)
            param.weight_loader(param, torch.full((6,), generation + 0.5))
        for index, name in enumerate(order):
            param = getattr(attention, name).weight
            source = torch.full(param.shape, generation * 4 + index, dtype=param.dtype)
            param.weight_loader(param, source)
            assert attention.get_buffer(_PACKED_CONV_WEIGHT_NAME).is_meta
            torch.testing.assert_close(captured, before, rtol=0, atol=0)

        with patch("vllm.model_executor.model_loader.reload.layerwise.logger.warning") as warning:
            finalize_layerwise_reload(model, config)
        warning.assert_not_called()

        assert events == list(order)
        assert attention._conv_weights_t(config.dtype) is captured
        assert _PACKED_CONV_WEIGHT_NAME in attention._non_persistent_buffers_set
        assert {name: tensor.data_ptr() for name, tensor in attention.named_parameters()} == original_ptrs
        torch.testing.assert_close(captured, _expected(attention, config.dtype), rtol=0, atol=0)
        assert not torch.equal(captured, before)
        for name in ("A_log", "dt_bias"):
            torch.testing.assert_close(getattr(attention, name), torch.full((6,), generation + 0.5))


@torch.no_grad()
def test_partial_rl_update_keeps_unloaded_children_and_repacks_after_restore():
    model, attention, config, events = _new_attention()
    process_weights_after_loading(model, config, torch.device("cpu"))
    captured = attention._conv_weights_t(config.dtype)
    q_before = attention.q_conv1d.weight.clone()
    v_before = attention.v_conv1d.weight.clone()
    events.clear()

    initialize_layerwise_reload(model)
    param = attention.k_conv1d.weight
    param.weight_loader(param, torch.full((6, 1, 4), -2.0))
    assert attention.q_conv1d.weight.is_meta and attention.v_conv1d.weight.is_meta
    finalize_layerwise_reload(model, config)

    assert events == ["k_conv1d"]
    assert attention._conv_weights_t(config.dtype) is captured
    torch.testing.assert_close(attention.q_conv1d.weight, q_before, rtol=0, atol=0)
    torch.testing.assert_close(attention.v_conv1d.weight, v_before, rtol=0, atol=0)
    torch.testing.assert_close(captured, _expected(attention, config.dtype), rtol=0, atol=0)


@torch.no_grad()
def test_prepared_weight_read_has_no_device_work_and_rejects_wrong_dtype():
    model, attention, config, _events = _new_attention()
    process_weights_after_loading(model, config, torch.device("cpu"))
    operations = []

    class RecordDispatch(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            operations.append(func)
            return func(*args, **(kwargs or {}))

    with RecordDispatch():
        first = attention._conv_weights_t(config.dtype)
        second = attention._conv_weights_t(config.dtype)
    assert first is second
    assert operations == []
    with pytest.raises(RuntimeError, match="same dtype"):
        attention._conv_weights_t(torch.float32)


@torch.no_grad()
def test_postload_rejects_incomplete_or_incompatible_storage():
    _model, attention, config, _events = _new_attention()
    with pytest.raises(RuntimeError, match="activation dtype"):
        attention.process_weights_after_loading(torch.float32)
    attention.q_conv1d.weight = nn.Parameter(torch.empty(6, 1, 4, device="meta"))
    with pytest.raises(RuntimeError, match="all be loaded"):
        attention.process_weights_after_loading(config.dtype)
    attention.q_conv1d.weight = nn.Parameter(torch.empty(5, 1, 4))
    with pytest.raises(RuntimeError, match="shapes cannot change"):
        attention.process_weights_after_loading(config.dtype)
    assert not attention._ascend_conv_weights_ready


@torch.no_grad()
def test_parent_postload_is_preserved_before_conv_packing():
    _model, attention, config, _events = _new_attention()

    def parent_postload(instance, dtype):
        assert instance is attention and dtype == config.dtype
        instance.q_conv1d.weight.add_(10)

    with patch.object(BailingMoeV3KimiDeltaAttention, "process_weights_after_loading", parent_postload, create=True):
        attention.process_weights_after_loading(config.dtype)
    torch.testing.assert_close(
        attention._conv_weights_t(config.dtype), _expected(attention, config.dtype), rtol=0, atol=0
    )


@torch.no_grad()
def test_derived_buffer_never_changes_strict_checkpoint_weight_tracking():
    model, attention, config, _events = _new_attention()
    loader = DefaultModelLoader.__new__(DefaultModelLoader)
    expected_names = {"0.A_log", "0.dt_bias", *(f"0.{name}.weight" for name in _CONVS)}

    def check():
        assert set(dict(model.named_parameters())) == expected_names
        assert f"0.{_PACKED_CONV_WEIGHT_NAME}" not in model.state_dict()
        loader.track_weights_loading(model, expected_names.copy())
        # Do not accidentally exempt actual parent checkpoint weights.
        with pytest.raises(ValueError, match="A_log"):
            loader.track_weights_loading(model, expected_names - {"0.A_log"})

    check()  # Default loader checks before any postload processing.
    process_weights_after_loading(model, config, torch.device("cpu"))
    captured = attention._conv_weights_t(config.dtype)
    check()
    for generation in (1, 2):
        initialize_layerwise_reload(model)
        check()
        for name in _CONVS:
            weight = getattr(attention, name).weight
            weight.weight_loader(weight, torch.full(weight.shape, generation, dtype=weight.dtype))
        finalize_layerwise_reload(model, config)
        check()
        assert attention._conv_weights_t(config.dtype) is captured
        torch.testing.assert_close(captured, _expected(attention, config.dtype), rtol=0, atol=0)


@torch.no_grad()
def test_first_postload_follows_child_device_but_later_reloads_cannot_move_storage():
    # Indexed CPU devices stay distinct under FakeTensorMode without requiring
    # an accelerator backend. The NPU test covers actual CPU-to-NPU movement.
    _model, attention, config, _events = _new_attention()
    with FakeTensorMode(allow_non_fake_inputs=True):
        initial = attention.get_buffer(_PACKED_CONV_WEIGHT_NAME)
        for name in _CONVS:
            child = getattr(attention, name)
            child.weight = nn.Parameter(torch.empty_like(child.weight, device="cpu:1"), requires_grad=False)
        attention.process_weights_after_loading(config.dtype)
        captured = attention._conv_weights_t(config.dtype)
        assert captured is not initial
        assert captured.device == torch.device("cpu:1")
        assert _PACKED_CONV_WEIGHT_NAME in attention._non_persistent_buffers_set
        attention.process_weights_after_loading(config.dtype)
        assert attention._conv_weights_t(config.dtype) is captured

        for name in _CONVS:
            child = getattr(attention, name)
            child.weight = nn.Parameter(torch.empty_like(child.weight, device="cpu:2"), requires_grad=False)
        # A failed reload clears readiness, but must not unlock graph storage.
        for _ in range(2):
            with pytest.raises(RuntimeError, match="cannot change device"):
                attention.process_weights_after_loading(config.dtype)
            assert attention.get_buffer(_PACKED_CONV_WEIGHT_NAME) is captured
            assert not attention._ascend_conv_weights_ready
