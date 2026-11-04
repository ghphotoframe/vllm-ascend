# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.ops.fused_moe import moe_comm_method as comm_module
from vllm_ascend.ops.fused_moe import routed_experts as routed_module
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEWeights
from vllm_ascend.ops.fused_moe.moe_comm_method import FusedMC2CommImpl
from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithMC2
from vllm_ascend.quantization.quant_type import QuantType


def make_layer(intermediate=768, dtype=torch.bfloat16):
    generator = torch.Generator().manual_seed(17)
    return SimpleNamespace(
        layer_name="model.layers.2.mlp.experts",
        moe_config=SimpleNamespace(intermediate_size_per_partition=intermediate),
        w13_weight=nn.Parameter(torch.randn(2, 2 * intermediate, 8, generator=generator).to(dtype)),
        w2_weight=nn.Parameter(torch.randn(2, 8, intermediate, generator=generator).to(dtype)),
    )


def make_method():
    method = object.__new__(AscendUnquantizedFusedMoEMethod)
    method.moe = SimpleNamespace(has_bias=False, is_act_and_mul=True)
    method.dynamic_eplb = False
    method._maybe_pad_weight = lambda weight: weight
    return method


@pytest.fixture
def a3_profile(monkeypatch):
    profile = get_hardware_profile(AscendDeviceType.A3)
    monkeypatch.setattr(routed_module, "get_current_hardware_profile", lambda: profile)
    monkeypatch.setattr(comm_module, "get_current_hardware_profile", lambda: profile)
    monkeypatch.setattr(routed_module, "get_ascend_config", lambda: SimpleNamespace(enable_megamoe_weight_padding=True))


@pytest.mark.parametrize("limit", [None, 4.0, 10.0])
def test_padding_preserves_each_projection_and_mlp_math(a3_profile, limit):
    layer = make_layer()
    original_w13, original_w2 = layer.w13_weight.detach(), layer.w2_weight.detach()
    w13, w2 = make_method()._align_cann_megamoe_weights(layer, use_megamoe=True)

    assert w13.shape == (2, 2048, 8)
    assert w2.shape == (2, 8, 1024)
    assert w13.dtype == w2.dtype == torch.bfloat16
    torch.testing.assert_close(w13[:, :768], original_w13[:, :768], rtol=0, atol=0)
    torch.testing.assert_close(w13[:, 1024:1792], original_w13[:, 768:], rtol=0, atol=0)
    torch.testing.assert_close(w2[:, :, :768], original_w2, rtol=0, atol=0)
    assert torch.count_nonzero(w13[:, 768:1024]) == 0
    assert torch.count_nonzero(w13[:, 1792:]) == 0
    assert torch.count_nonzero(w2[:, :, 768:]) == 0

    # FP64 checks the mathematical transform, not CANN's BF16 rounding.
    x = torch.linspace(-2, 2, 24, dtype=torch.float64).reshape(3, 8)

    def mlp(gate_up, down):
        gate, up = (x @ gate_up.double().transpose(1, 2)).chunk(2, dim=-1)
        gate = F.silu(gate)
        if limit is not None:
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        return (gate * up) @ down.double().transpose(1, 2)

    torch.testing.assert_close(mlp(w13, w2), mlp(original_w13, original_w2), rtol=1e-12, atol=1e-12)
    assert layer.moe_config.intermediate_size_per_partition == 768
    assert layer.w13_weight.shape == (2, 1536, 8)


@pytest.mark.parametrize("intermediate", [512, 1024, 1536])
def test_aligned_weights_keep_original_storage(a3_profile, intermediate):
    layer = make_layer(intermediate=intermediate)
    w13, w2 = make_method()._align_cann_megamoe_weights(layer, use_megamoe=True)
    assert w13.data_ptr() == layer.w13_weight.data_ptr()
    assert w2.data_ptr() == layer.w2_weight.data_ptr()


def test_disabled_padding_keeps_unaligned_weights_and_storage(monkeypatch, a3_profile):
    layer = make_layer()
    monkeypatch.setattr(
        routed_module, "get_ascend_config", lambda: SimpleNamespace(enable_megamoe_weight_padding=False)
    )
    w13, w2 = make_method()._align_cann_megamoe_weights(layer, use_megamoe=True)
    assert w13.shape == (2, 1536, 8)
    assert w2.shape == (2, 8, 768)
    assert w13.data_ptr() == layer.w13_weight.data_ptr()
    assert w2.data_ptr() == layer.w2_weight.data_ptr()


@pytest.mark.parametrize(
    ("device", "dtype", "enabled"),
    [
        (AscendDeviceType.A3, torch.bfloat16, False),
        (AscendDeviceType.A2, torch.bfloat16, True),
        (AscendDeviceType.A5, torch.bfloat16, True),
        (AscendDeviceType.A3, torch.float16, True),
        (AscendDeviceType.A3, torch.float32, True),
    ],
)
def test_padding_leaves_other_paths_unchanged(monkeypatch, device, dtype, enabled):
    monkeypatch.setattr(routed_module, "get_current_hardware_profile", lambda: get_hardware_profile(device))
    layer = make_layer(dtype=dtype)
    w13, w2 = make_method()._align_cann_megamoe_weights(layer, use_megamoe=enabled)
    assert w13.data_ptr() == layer.w13_weight.data_ptr()
    assert w2.data_ptr() == layer.w2_weight.data_ptr()


@pytest.mark.parametrize("invalid", ["bias", "ungated", "shape"])
def test_padding_rejects_unsupported_weight_layout(a3_profile, invalid):
    method, layer = make_method(), make_layer()
    if invalid == "bias":
        method.moe.has_bias = True
    elif invalid == "ungated":
        method.moe.is_act_and_mul = False
    else:
        layer.w13_weight = nn.Parameter(layer.w13_weight[:, :-1])
    with pytest.raises(ValueError, match="A3 BF16 MegaMoE"):
        method._align_cann_megamoe_weights(layer, use_megamoe=True)


@pytest.mark.parametrize("padding_enabled", [True, False])
def test_postload_produces_contiguous_expert_lists(monkeypatch, a3_profile, padding_enabled):
    method, layer = make_method(), make_layer()
    original_w13, original_w2 = layer.w13_weight.detach().clone(), layer.w2_weight.detach().clone()
    monkeypatch.setattr(
        routed_module,
        "get_ascend_config",
        lambda: SimpleNamespace(enable_fused_mc2=1, enable_megamoe_weight_padding=padding_enabled),
    )
    monkeypatch.setattr(routed_module, "get_current_vllm_config", lambda: None)
    monkeypatch.setattr(routed_module, "use_cann_megamoe", lambda _: True)
    monkeypatch.setattr(routed_module.torch.npu, "empty_cache", lambda: None)
    monkeypatch.setattr(
        AscendUnquantizedFusedMoEMethod.__mro__[2], "process_weights_after_loading", lambda *_: None, raising=False
    )

    method.process_weights_after_loading(layer)

    assert not hasattr(layer, "w13_weight")
    assert not hasattr(layer, "w2_weight")
    assert layer.moe_config.intermediate_size_per_partition == 768
    assert len(layer.w13_weight_list) == len(layer.w2_weight_list) == 2
    physical_intermediate = 1024 if padding_enabled else 768
    for expert, (w13, w2) in enumerate(zip(layer.w13_weight_list, layer.w2_weight_list)):
        assert w13.is_contiguous() and w2.is_contiguous()
        assert w13.shape == (8, 2 * physical_intermediate)
        assert w2.shape == (physical_intermediate, 8)
        torch.testing.assert_close(w13[:, :768], original_w13[expert, :768].T, rtol=0, atol=0)
        torch.testing.assert_close(
            w13[:, physical_intermediate : physical_intermediate + 768], original_w13[expert, 768:].T, rtol=0, atol=0
        )
        torch.testing.assert_close(w2[:768], original_w2[expert].T, rtol=0, atol=0)
    # EPLB sees the same per-expert lists; this does not exercise live migration.
    views = method.get_eplb_weight_views(layer)
    assert views[0] is layer.w13_weight_list
    assert views[1] is layer.w2_weight_list


@pytest.mark.parametrize(
    "dtype, quant_type, expected_physical",
    [
        (torch.bfloat16, QuantType.NONE, 1024),
        (torch.int8, QuantType.W8A8, None),
        (torch.uint8, QuantType.W4A8MXFP, None),
    ],
)
def test_cann_initialization_uses_physical_bf16_intermediate(
    monkeypatch, a3_profile, dtype, quant_type, expected_physical
):
    comm = object.__new__(FusedMC2CommImpl)
    comm.token_dispatcher = object.__new__(TokenDispatcherWithMC2)
    comm.token_dispatcher.max_num_tokens_per_rank = 1024
    comm.token_dispatcher.global_bs = 0
    comm.mega_moe_symm_buffer = None
    comm._init_mega_moe_symm_buffer = MagicMock(return_value=object())
    comm.mega_moe = MagicMock(return_value=(torch.zeros(1, 8), torch.zeros(2)))
    monkeypatch.setattr(comm_module.moe_utils, "_get_cann_mega_moe_quant_settings", lambda _: (0, None, None))
    inp = SimpleNamespace(
        hidden_states=torch.zeros(1, 8, dtype=torch.bfloat16),
        topk_ids=torch.zeros(1, 2, dtype=torch.int32),
        topk_weights=torch.ones(1, 2),
        quant=SimpleNamespace(quant_type=quant_type),
        routing=SimpleNamespace(mc2_mask=None),
        layer=SimpleNamespace(mega_moe_activation_kwargs={}),
    )
    if dtype == torch.uint8:
        # A5 packed FP4 [2I, H/2] and [H, I/2] must not be interpreted
        # like the transposed, unpacked BF16 [H, 2I] and [I, H] layouts.
        weights = MoEWeights(w1=[torch.zeros(2048, 4, dtype=dtype)], w2=[torch.zeros(8, 512, dtype=dtype)])
    else:
        weights = MoEWeights(w1=[torch.zeros(8, 2048, dtype=dtype)], w2=[torch.zeros(1024, 8, dtype=dtype)])
    comm._apply_cann_mega_moe(inp, weights, is_decode_only_node=False)
    assert comm._init_mega_moe_symm_buffer.call_args.kwargs["intermediate_hidden"] == expected_physical


@pytest.mark.parametrize("physical, expected", [(None, 768), (1024, 1024)])
def test_symm_buffer_receives_physical_size_without_mutating_config(monkeypatch, physical, expected):
    comm = object.__new__(FusedMC2CommImpl)
    comm.moe_config = SimpleNamespace(
        experts_per_token=8, num_experts=512, hidden_dim=2560, intermediate_size_per_partition=768
    )
    comm.token_dispatcher = object.__new__(TokenDispatcherWithMC2)
    comm.token_dispatcher.max_num_tokens_per_rank = 1024
    comm.token_dispatcher.global_bs = 0
    comm.token_dispatcher.ep_world_size = 8
    comm.get_symm_buffer_for_mega_moe = MagicMock()
    monkeypatch.setattr(comm_module, "get_mc2_group", lambda: SimpleNamespace(device_group="ep8"))
    comm._init_mega_moe_symm_buffer(is_decode_only_node=True, intermediate_hidden=physical)
    kwargs = comm.get_symm_buffer_for_mega_moe.call_args.kwargs
    assert kwargs["intermediate_hidden"] == expected
    assert kwargs["max_recv_token_num"] == 65536
    assert comm.moe_config.intermediate_size_per_partition == 768
