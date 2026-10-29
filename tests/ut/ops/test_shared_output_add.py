# SPDX-License-Identifier: Apache-2.0
"""Single-pair shared/routed output joins preserve the runner contract."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm_ascend.ops.fused_moe import fused_moe as fused_moe_module
from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner


def _runner(monkeypatch, shared, routed, *, super_kernel=False, release_schedule=False, reduced=True):
    runner = AscendMoERunner.__new__(AscendMoERunner)
    nn.Module.__init__(runner)
    runner._release_shared_schedule = release_schedule
    runner.apply_routed_input_transform = lambda states: (states, states)
    runner._maybe_pad_hidden_states = lambda *_: (routed, None, None)
    runner._forward_entry = lambda *_: (shared, routed)
    runner._encode_layer_name = lambda: "test.shared_output_add"
    monkeypatch.setattr(
        AscendMoERunner, "_quant_method", property(lambda _: SimpleNamespace(has_unpadded_output=False))
    )
    monkeypatch.setattr(AscendMoERunner, "_fused_output_is_reduced", property(lambda _: reduced))
    monkeypatch.setattr(fused_moe_module, "_unpack", lambda outputs: outputs)
    monkeypatch.setattr(
        fused_moe_module,
        "get_ascend_config",
        lambda: SimpleNamespace(ascend_compilation_config=SimpleNamespace(enable_super_kernel=super_kernel)),
    )
    runner._maybe_reduce_routed_output_before_transform = lambda output, is_reduced: (output, is_reduced)
    runner._maybe_reduce_shared_expert_output = lambda output, _: output
    runner._maybe_apply_routed_scale_to_output = lambda shared, routed: (shared, routed)
    runner.apply_routed_output_transform = lambda output: output
    runner._maybe_reduce_final_output = lambda output, *_: output
    runner._maybe_add_zero_expert_output = lambda output: output
    return runner


@pytest.mark.parametrize(
    "batch,dtype,strided",
    [
        (1, torch.bfloat16, False),
        (128, torch.bfloat16, False),
        (4, torch.bfloat16, True),
        (4, torch.float16, False),
        (4, torch.float32, False),
    ],
)
def test_shared_output_add_preserves_inputs_and_allocates_output(monkeypatch, batch, dtype, strided):
    width = 32
    shared = torch.arange(batch * width, dtype=torch.float32).reshape(batch, width).remainder(17).to(dtype)
    routed = shared.flip(-1).neg()
    if strided:
        shared, routed = shared[:, ::2], routed[:, ::2]
        assert not shared.is_contiguous() and not routed.is_contiguous()
    shared_before, routed_before = shared.clone(), routed.clone()
    expected = torch._foreach_add([shared], [routed])[0]
    runner = _runner(monkeypatch, shared, routed)
    foreach = Mock(side_effect=AssertionError("Ordinary graphs must use the single-pair Add"))
    monkeypatch.setattr(torch, "_foreach_add", foreach)

    output = runner.forward(routed, routed, shared_experts_input=shared)

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert output.dtype == dtype
    assert output.untyped_storage().data_ptr() not in {
        shared.untyped_storage().data_ptr(),
        routed.untyped_storage().data_ptr(),
    }
    output.zero_()
    torch.testing.assert_close(shared, shared_before, rtol=0, atol=0)
    torch.testing.assert_close(routed, routed_before, rtol=0, atol=0)
    foreach.assert_not_called()


@pytest.mark.parametrize("super_kernel", [False, True])
@pytest.mark.parametrize("release_schedule", [False, True])
def test_shared_output_add_keeps_superkernel_and_release_schedule(monkeypatch, super_kernel, release_schedule):
    shared, routed = torch.full((2, 4), 3.0), torch.full((2, 4), 5.0)
    runner = _runner(monkeypatch, shared, routed, super_kernel=super_kernel, release_schedule=release_schedule)
    foreach = Mock(wraps=torch._foreach_add)
    monkeypatch.setattr(torch, "_foreach_add", foreach)

    output = runner.forward(routed, routed, shared_experts_input=shared)

    torch.testing.assert_close(output, shared + routed, rtol=0, atol=0)
    assert foreach.call_count == int(super_kernel and not release_schedule)


def test_shared_output_add_is_skipped_without_shared_experts(monkeypatch):
    routed = torch.randn(2, 4)
    runner = _runner(monkeypatch, None, routed)
    monkeypatch.setattr(
        fused_moe_module, "get_ascend_config", Mock(side_effect=AssertionError("No shared output to combine"))
    )
    assert runner.forward(routed, routed) is routed


@pytest.mark.parametrize("initially_reduced", [False, True])
def test_shared_output_add_preserves_reduce_scale_transform_order(monkeypatch, initially_reduced):
    shared, routed = torch.full((2, 4), 2.0), torch.full((2, 4), 3.0)
    runner = _runner(monkeypatch, shared, routed, reduced=initially_reduced)
    order = []

    def reduce_routed(output, reduced):
        order.append("reduce_routed")
        assert reduced is initially_reduced
        return (output if reduced else output * 2), True

    def reduce_shared(output, reduced):
        order.append("reduce_shared")
        assert reduced
        return output * 2

    def scale(shared, routed):
        order.append("scale")
        return shared, routed * 3

    def transform(output):
        order.append("transform")
        return output.square()

    expected = shared * 2 + (routed * (1 if initially_reduced else 2) * 3).square()

    def final_reduce(output, unpadded_dim, reduced):
        order.append("final_reduce")
        assert unpadded_dim is None and reduced
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        return output

    def zero_experts(output):
        order.append("zero_experts")
        return output + 1

    runner._maybe_reduce_routed_output_before_transform = reduce_routed
    runner._maybe_reduce_shared_expert_output = reduce_shared
    runner._maybe_apply_routed_scale_to_output = scale
    runner.apply_routed_output_transform = transform
    runner._maybe_reduce_final_output = final_reduce
    runner._maybe_add_zero_expert_output = zero_experts

    output = runner.forward(routed, routed, shared_experts_input=shared)

    torch.testing.assert_close(output, expected + 1, rtol=0, atol=0)
    assert order == ["reduce_routed", "reduce_shared", "scale", "transform", "final_reduce", "zero_experts"]
