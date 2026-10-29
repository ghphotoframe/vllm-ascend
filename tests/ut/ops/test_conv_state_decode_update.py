# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata/launch contracts for the shared single-token state helper."""

from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.ops import bailing_moe_v3_kda as bailing_kda
from vllm_ascend.ops import kda_conv
from vllm_ascend.ops.triton.kda import conv_state


def _inputs(requests=2, state_len=6, dim=513, layout="sd"):
    span = state_len * dim
    backing = torch.empty(5 * (span + 32) + 17, dtype=torch.float32)
    stride = (span + 32, dim, 1) if layout == "sd" else (span + 32, 1, state_len)
    state = backing.as_strided((5, state_len, dim), stride, 17)
    return [
        state,
        torch.empty(requests, state_len, dim, dtype=torch.bfloat16),
        torch.empty(requests, dim, dtype=torch.bfloat16),
        torch.empty(requests * 2, dtype=torch.int64)[::2],
        torch.arange((requests + 1) * 2, dtype=torch.int64)[::2],
        torch.empty(requests, dtype=torch.int32),
    ]


@pytest.mark.parametrize("layout", ["sd", "ds"])
@pytest.mark.parametrize("state_len", [3, 6, 32])
def test_one_program_owns_all_history_rows_per_channel_tile(monkeypatch, layout, state_len):
    args = _inputs(layout=layout, state_len=state_len)
    kernel = MagicMock()
    monkeypatch.setattr(conv_state, "_gather_update_conv_state_kernel", kernel)
    conv_state.gather_update_conv_state(*args)
    kernel.__getitem__.assert_called_once_with((6,))
    call = kernel.__getitem__.return_value.call_args
    assert all(actual is expected for actual, expected in zip(call.args[:6], args))
    assert call.args[6:12] == (args[0].stride(0), 2, 2, 513, 5, 2)
    assert call.args[12:] == (2, state_len, 513, args[0].stride(1), args[0].stride(2))
    assert call.kwargs == {"NULL_SLOT": 0, "BLOCK": 256}


def test_zero_requests_does_not_launch(monkeypatch):
    kernel = MagicMock()
    monkeypatch.setattr(conv_state, "_gather_update_conv_state_kernel", kernel)
    conv_state.gather_update_conv_state(*_inputs(requests=0))
    kernel.__getitem__.assert_not_called()


@pytest.mark.parametrize(
    "case",
    [
        "dtype",
        "packed_dtype",
        "packed_stride",
        "indices",
        "boundaries",
        "history",
        "overlap",
        "channel_slice",
        "fewer_input_rows",
        "extra_input_rows",
    ],
)
def test_unsupported_contract_rejected_before_launch(monkeypatch, case):
    args = _inputs()
    if case == "dtype":
        args[2] = args[2].float()
    elif case == "packed_dtype":
        args[1] = args[1].float()
    elif case == "packed_stride":
        args[1] = torch.empty(4, 6, 513, dtype=torch.bfloat16)[::2]
    elif case == "indices":
        args[3] = args[3].view(2, 1)
    elif case == "boundaries":
        args[4] = args[4][:-1]
    elif case == "history":
        args = _inputs(state_len=2)
    elif case == "channel_slice":
        args[0] = torch.empty(5, 6, 1026)[:, :, ::2]
    elif case == "overlap":
        args[0] = args[0].as_strided(args[0].shape, (1, 513, 1))
    elif case == "fewer_input_rows":
        args[2] = args[2][:1]
    elif case == "extra_input_rows":
        args[2] = torch.empty(3, 513, dtype=torch.bfloat16)
    kernel = MagicMock()
    monkeypatch.setattr(conv_state, "_gather_update_conv_state_kernel", kernel)
    with pytest.raises(ValueError):
        conv_state.gather_update_conv_state(*args)
    kernel.__getitem__.assert_not_called()


@pytest.mark.parametrize("case", ["channel_slice", "compact_varlen"])
def test_unsupported_fastpath_falls_back_to_two_selected_copies(monkeypatch, case):
    """Keep the existing stride and compact-input contracts in the fallback."""
    cls = bailing_kda.AscendBailingMoeV3KimiDeltaAttention
    attention = cls.__new__(cls)
    torch.nn.Module.__init__(attention)
    attention._conv_state_dim_first = False
    attention.num_speculative_tokens = 0
    backing = torch.randn(7 * 52 + 7, dtype=torch.float32)
    stride = (52, 12, 2) if case == "channel_slice" else (52, 6, 1)
    state = backing.as_strided((7, 3, 6), stride, 7)
    expected = backing.clone()
    expected_state = expected.as_strided(state.shape, state.stride(), state.storage_offset())
    x = torch.randn(2 if case == "channel_slice" else 1, 6, dtype=torch.bfloat16)
    weight = torch.randn(4, 6, dtype=torch.bfloat16)
    indices = torch.tensor([1, 4], dtype=torch.int32)
    starts = torch.tensor([0, 1, 2] if case == "channel_slice" else [0, 0, 1], dtype=torch.int32)
    result = torch.ones_like(x)
    for request, slot in enumerate(indices.tolist()):
        if starts[request + 1] == starts[request]:
            continue
        old = state[slot].bfloat16()
        expected_state[slot] = torch.stack((old[1], old[2], x[starts[request]])).float()

    def copy_selected(cache, packed, selected, boundaries, packed_indices, *, write_back):
        assert cache is state and boundaries is starts
        for request, slot in enumerate(selected.tolist()):
            if boundaries[request + 1] == boundaries[request]:
                if not write_back:
                    packed[request].zero_()
                    packed_indices[request] = -1
                continue
            if write_back:
                cache[slot].copy_(packed[request])
            else:
                packed[request].copy_(cache[slot])
                packed_indices[request] = request

    def native_update(input_, packed, weight_, **kwargs):
        assert input_ is x and weight_ is weight
        assert kwargs["num_accepted_tokens"] is None
        assert kwargs["max_query_len"] == 1
        assert kwargs["null_block_id"] == 2
        for request, slot in enumerate(kwargs["conv_state_indices"].tolist()):
            if slot == 2:
                continue
            old = packed[slot].clone()
            packed[slot] = torch.stack((old[1], old[2], x[starts[request]]))
        return result

    copies = MagicMock(side_effect=copy_selected)
    fast = MagicMock(side_effect=AssertionError("Unsupported layout/input shape entered fast path"))
    monkeypatch.setattr(bailing_kda, "copy_conv_state", copies)
    monkeypatch.setattr(bailing_kda, "gather_update_conv_state", fast)
    monkeypatch.setattr(kda_conv, "causal_conv1d_update", native_update)
    actual = attention._run_causal_conv1d(x, weight, state, starts, indices, None, run_mode=1, max_query_len=1)
    assert actual is result
    fast.assert_not_called()
    assert copies.call_count == 2
    assert [call.kwargs["write_back"] for call in copies.call_args_list] == [False, True]
    torch.testing.assert_close(backing, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("row_gap", [0, 5])
def test_native_pagegap_preserves_storage_and_masks_inactive_requests(monkeypatch, dtype, row_gap):
    cls = bailing_kda.AscendBailingMoeV3KimiDeltaAttention
    attention = cls.__new__(cls)
    torch.nn.Module.__init__(attention)
    attention._conv_state_dim_first = False
    attention.num_speculative_tokens = 0
    stride = (64, 6 + row_gap, 1)
    backing = torch.arange(7 * 64, dtype=dtype)
    state = backing.as_strided((6, 3, 6), stride, 9)
    expected = backing.clone()
    x = torch.full((5, 6), 0.5, dtype=dtype)
    weight = torch.ones(4, 6, dtype=dtype)
    indices = torch.tensor([[2], [3], [-1], [6], [0]], dtype=torch.int32)
    starts = torch.tensor([0, 1, 1, 2, 3, 4], dtype=torch.int32)
    expected_state = expected.as_strided(state.shape, stride, 9)
    expected_state[2].copy_(torch.cat((state[2, 1:].clone(), x[:1]), dim=0))

    def native_update(input_, cache, weight_, **kwargs):
        assert input_ is x and cache is state and weight_ is weight
        assert kwargs["conv_state_indices"].tolist() == [2, 0, 0, 0, 0]
        assert kwargs["null_block_id"] == 0
        assert torch.count_nonzero(kwargs["out"]) == 0
        cache[2].copy_(torch.cat((cache[2, 1:].clone(), input_[:1]), dim=0))
        kwargs["out"][0].fill_(2)
        return kwargs["out"]

    fail_copy = MagicMock(side_effect=AssertionError("Native-compatible state was staged"))
    monkeypatch.setattr(bailing_kda, "copy_conv_state", fail_copy)
    monkeypatch.setattr(bailing_kda, "gather_update_conv_state", fail_copy)
    monkeypatch.setattr(kda_conv, "causal_conv1d_update", native_update)
    actual = attention._run_causal_conv1d(x, weight, state, starts, indices, None, run_mode=1, max_query_len=1)
    assert torch.count_nonzero(actual[1:]) == 0
    torch.testing.assert_close(backing, expected, rtol=0, atol=0)
    fail_copy.assert_not_called()
