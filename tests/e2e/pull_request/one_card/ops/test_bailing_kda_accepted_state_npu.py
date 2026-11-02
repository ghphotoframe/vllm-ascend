# SPDX-License-Identifier: Apache-2.0
"""Regression for previous acceptance larger than the current verify length."""

import itertools

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.kda import run_recurrent_kda
from vllm_ascend.ops.triton.kda.accepted_state import rebase_short_kda_queries

_HEADS = 8
_HEAD_DIM = 128
_SLOTS = 8


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU hardware required")
@pytest.mark.parametrize("padding,length,accepted", itertools.product([0, 64], range(1, 5), range(1, 5)))
def test_short_verify_matches_explicitly_selected_prior_state(padding, length, accepted):
    torch.npu.set_device(0)
    torch.manual_seed(17)
    payload = _HEADS * _HEAD_DIM * _HEAD_DIM
    backing = torch.zeros(_SLOTS * (payload + padding), dtype=torch.float32, device="npu")
    state = torch.as_strided(
        backing, (_SLOTS, _HEADS, _HEAD_DIM, _HEAD_DIM), (payload + padding, _HEAD_DIM * _HEAD_DIM, _HEAD_DIM, 1)
    )
    for slot in range(1, 5):
        state[slot].fill_(slot / 32)
    reference = torch.zeros_like(state)
    reference[1].copy_(state[accepted])
    inputs = [(torch.randn(1, length, _HEADS, _HEAD_DIM) * 0.2).to("npu", torch.bfloat16) for _ in range(4)]
    beta = torch.full((1, length, _HEADS), 0.5, device="npu")
    a_log = torch.zeros(_HEADS, device="npu")
    bias = torch.zeros(_HEADS, _HEAD_DIM, device="npu")
    slots = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32, device="npu")
    starts = torch.tensor([0, length], dtype=torch.int32, device="npu")
    counts = torch.tensor([accepted], dtype=torch.int32, device="npu")
    adjusted = rebase_short_kda_queries(state, slots, starts, counts)
    actual = run_recurrent_kda(
        *inputs, beta, state, starts, slots, a_log, bias, lower_bound=-5.0, num_accepted_tokens=adjusted
    )
    expected = run_recurrent_kda(
        *inputs,
        beta,
        reference,
        starts,
        slots,
        a_log,
        bias,
        lower_bound=-5.0,
        num_accepted_tokens=torch.ones_like(counts),
    )
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(state[1 : 1 + length], reference[1 : 1 + length], rtol=0, atol=0)
    assert counts.cpu().tolist() == [accepted]
