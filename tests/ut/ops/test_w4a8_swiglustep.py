# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.quantization.methods.w4a8 import w4a8


@pytest.mark.parametrize("layout", ["per_channel", "expert_list", "prepared_group"])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("group_list_type", [0, 1])
@pytest.mark.parametrize("scale_dtype", [torch.float32, torch.int64])
def test_generic_gmm1_uses_all_scales_and_preserves_msd_and_bf16(
    monkeypatch, layout, has_bias, group_list_type, scale_dtype
):
    method = object.__new__(w4a8.AscendW4A8DynamicFusedMoEMethod)
    method.use_expert_weight_list = layout == "expert_list"
    experts, hidden, outputs, tokens = 2, 16, 8, 4
    scale_shape = (experts, 2, outputs) if layout == "prepared_group" else (experts, outputs)
    layer = SimpleNamespace(
        w13_weight=torch.zeros((experts, hidden, outputs // 8), dtype=torch.int32),
        w2_weight=torch.zeros((experts, outputs // 2, hidden // 8), dtype=torch.int32),
        w13_weight_scale=torch.arange(torch.Size(scale_shape).numel(), dtype=scale_dtype).reshape(scale_shape),
        w2_weight_scale=torch.ones((experts, 1, hidden), dtype=torch.int64),
        w13_scale_bias=torch.arange(experts * outputs, dtype=torch.float32).reshape(experts, outputs),
        w2_scale_bias=torch.ones((experts, hidden)),
    )
    if not has_bias:
        layer.w13_scale_bias = layer.w2_scale_bias = None
    if method.use_expert_weight_list:
        for name in (
            "w13_weight",
            "w2_weight",
            "w13_weight_scale",
            "w2_weight_scale",
            "w13_scale_bias",
            "w2_scale_bias",
        ):
            value = getattr(layer, name)
            setattr(layer, name + "_list", list(value.unbind(0)) if value is not None else None)

    hidden_states = torch.ones((tokens, hidden), dtype=torch.int8)
    token_scale = torch.linspace(0.25, 1.0, tokens)
    compute_input = SimpleNamespace(
        layer=layer,
        hidden_states=hidden_states,
        dynamic_scale=token_scale,
        activation=MoEActivation.SWIGLUSTEP,
        group_list=torch.tensor([1, 4] if group_list_type == 0 else [1, 3], dtype=torch.int32),
        group_list_type=group_list_type,
    )
    output = torch.ones((tokens, outputs), dtype=torch.bfloat16)
    events = []

    def grouped_matmul(**kwargs):
        events.append("gmm")
        assert kwargs["x"][0] is hidden_states
        assert kwargs["per_token_scale"][0] is token_scale
        assert kwargs["output_dtype"] == torch.bfloat16
        assert (kwargs["split_item"], kwargs["group_type"], kwargs["group_list_type"]) == (2, 0, 1)
        torch.testing.assert_close(kwargs["group_list"], torch.tensor([1, 3], dtype=torch.int32))
        weights, scales, _, _ = method._get_mlp_weights(layer)
        biases, _ = method._get_mlp_scale_bias(layer)
        assert len(kwargs["scale"]) == len(weights) == (experts if layout == "expert_list" else 1)
        for actual_weight, weight, actual_scale, scale in zip(kwargs["weight"], weights, kwargs["scale"], scales):
            assert actual_weight.data_ptr() == weight.data_ptr()
            expected = scale.to(torch.int64)
            if layout != "prepared_group":
                expected = expected.unsqueeze(-2)
            torch.testing.assert_close(actual_scale, expected, rtol=0, atol=0)
            if scale_dtype == torch.int64:
                assert actual_scale.data_ptr() == scale.data_ptr()
                if layout == "prepared_group":
                    assert actual_scale is scale
        if has_bias:
            assert len(kwargs["bias"]) == len(biases)
            assert all(actual is expected for actual, expected in zip(kwargs["bias"], biases))
        else:
            assert kwargs["bias"] is None
        return [output]

    def dispose(tensor):
        assert tensor is hidden_states
        assert events == ["gmm"]
        events.append("dispose")

    gmm = MagicMock(side_effect=grouped_matmul)
    monkeypatch.setattr(w4a8.torch_npu, "npu_grouped_matmul", gmm)
    monkeypatch.setattr(w4a8, "dispose_tensor", dispose)

    assert method.apply_gmm1(compute_input) is output
    gmm.assert_called_once()
    assert events == ["gmm", "dispose"]


@pytest.mark.parametrize(
    "activation,expected",
    [
        (MoEActivation.SWIGLUSTEP, False),
        ("swiglustep", False),
        (MoEActivation.SILU, True),
        ("silu", True),
        (MoEActivation.SITU, True),
        ("situ", True),
    ],
)
def test_swiglustep_uses_generic_gmm_without_changing_fused_activations(activation, expected):
    method = object.__new__(w4a8.AscendW4A8DynamicFusedMoEMethod)
    assert method.supports_fused_activation(activation) is expected
    assert method.fused_activations == frozenset({"silu", "situ"})
