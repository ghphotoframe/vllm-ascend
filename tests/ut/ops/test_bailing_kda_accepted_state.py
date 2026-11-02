# SPDX-License-Identifier: Apache-2.0
"""CPU metadata/dispatch contract; numerical kernel regressions live in e2e."""

import itertools

import pytest
import torch

from vllm_ascend.ops.triton.kda import accepted_state as module


class _CPUCommit:
    def __getitem__(self, grid):
        def execute(
            state, slots, starts, counts, adjusted, stride, capacity, payload, row_stride, col_stride, columns, **kwargs
        ):
            assert grid[0] == counts.numel()
            assert stride == state.stride(0)
            assert row_stride == slots.stride(0) and col_stride == slots.stride(1)
            for row, count in enumerate(counts.tolist()):
                length = int(starts[row + 1] - starts[row])
                active = 0 < length < count <= min(columns, 8)
                active = active and all(0 <= int(v) < capacity for v in slots[row, :length])
                active = active and 0 <= int(slots[row, count - 1]) < capacity
                if active:
                    state[int(slots[row, 0])].copy_(state[int(slots[row, count - 1])])
                    adjusted[row] = 1
                else:
                    adjusted[row] = count

        return execute


@pytest.mark.parametrize("length,count", itertools.product(range(5), range(6)))
def test_acceptance_dispatch_and_unchanged_host_metadata(monkeypatch, length, count):
    monkeypatch.setattr(module, "_rebase_short_query_kernel", _CPUCommit())
    state = torch.arange(128, dtype=torch.float32).reshape(8, 1, 4, 4)
    before = state.clone()
    counts = torch.tensor([count], dtype=torch.int32)
    result = module.rebase_short_kda_queries(state, torch.tensor([[1, 2, 3, 4]]), torch.tensor([0, length]), counts)
    active = 0 < length < count <= 4
    if active:
        before[1] = before[count]
    assert torch.equal(state, before)
    assert result.tolist() == [1 if active else count]
    assert counts.tolist() == [count]


@pytest.mark.parametrize("slots", [[[1, -1, 3, 4]], [[1, 2, 3, -1]]])
def test_invalid_slots_remain_untouched(monkeypatch, slots):
    monkeypatch.setattr(module, "_rebase_short_query_kernel", _CPUCommit())
    state = torch.arange(128, dtype=torch.float32).reshape(8, 1, 4, 4)
    before = state.clone()
    result = module.rebase_short_kda_queries(state, torch.tensor(slots), torch.tensor([0, 2]), torch.tensor([4]))
    assert torch.equal(state, before) and result.tolist() == [4]


def test_metadata_dtype_is_rejected_before_state_write():
    state = torch.zeros(8, 1, 4, 4)
    with pytest.raises(ValueError, match="INT32 or INT64"):
        module.rebase_short_kda_queries(state, torch.ones(1, 4), torch.tensor([0, 1]), torch.tensor([4]))
