# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.ops.fused_moe.shared_experts import AscendSharedExperts


@pytest.mark.parametrize("tokens", [1, 5])
@pytest.mark.parametrize("limit", [0.0, 10.0])
def test_unaligned_shared_activation_preserves_contract(tokens, limit):
    obj = AscendSharedExperts.__new__(AscendSharedExperts)
    obj.multistream_overlap = False
    x = torch.linspace(-12, 12, tokens * 320).reshape(tokens, 320)
    gate, up = x.chunk(2, -1)
    if limit:
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    expected = torch.nn.functional.silu(gate) * up

    def activation(value):
        g, u = value.chunk(2, -1)
        if limit:
            g, u = g.clamp(max=limit), u.clamp(-limit, limit)
        return torch.nn.functional.silu(g) * u

    obj.layer = SimpleNamespace(act_fn=activation, down_proj=object())
    obj.part2 = Mock(return_value=expected)
    obj._wait_for_routed_stage = Mock()
    submitted = SimpleNamespace(gate_up=x, hidden_states=torch.ones(tokens, 4))
    milestones = SimpleNamespace(shared_activation_overlap_start=object(), routed_combine_start=object())
    with patch.object(torch.ops._C_ascend, "npu_swiglu_group_quant", create=True) as fused:
        result = obj._run_a8_mxfp_mlp(submitted, milestones)
    fused.assert_not_called()
    assert torch.equal(obj.part2.call_args.args[1], expected)
    assert torch.equal(result, expected)
    assert obj._wait_for_routed_stage.call_count == 2


@pytest.mark.parametrize("width", [384, 512])
@pytest.mark.parametrize("limit", [0.0, 10.0])
def test_aligned_shared_expert_keeps_fused_kernel(width, limit):
    obj = AscendSharedExperts.__new__(AscendSharedExperts)
    obj.situ_activation = None
    obj.swiglu_limit = limit
    obj.apply_activation = Mock(side_effect=AssertionError("Unexpected separate path"))
    obj._wait_for_routed_stage = Mock()
    obj.part2 = Mock(return_value="fused")
    x = torch.zeros(2, width)
    q, scale = torch.zeros(2, width // 2), torch.ones(2, 2)
    with patch.object(
        torch.ops._C_ascend, "npu_swiglu_group_quant", return_value=(q, scale, None), create=True
    ) as fused:
        result = obj._run_a8_mxfp_mlp(
            SimpleNamespace(gate_up=x, hidden_states=x),
            SimpleNamespace(shared_activation_overlap_start=None, routed_combine_start=None),
        )
    assert result == "fused"
    fused.assert_called_once()
    assert fused.call_args.kwargs["clamp_value"] == limit
    obj.apply_activation.assert_not_called()
