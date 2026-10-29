# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real FLA reads/writes of same-dtype SD cache views without staging.

The dense reference uses the same native convolution math, but independent
storage and CPU-normalized metadata. CPU history checks and whole-backing
comparisons also protect unselected slots, NULL, row/page padding and offsets.
Run on an NPU with the supported stable FLA backend; no native op is mocked.
"""

from unittest.mock import MagicMock

import pytest
import torch
import torch_npu  # noqa: F401
from fla_npu.ops.ascendc import causal_conv1d_fn, causal_conv1d_update

from vllm_ascend.ops import bailing_moe_v3_kda as bailing_kda


def _states(dtype, *, batch=8, state_len=6, layout="page"):
    slots, dim, offset = batch + 5, 1536, 32
    row_stride = dim + (32 if layout == "row_page" else 0)
    span = (state_len - 1) * row_stride + dim
    page_stride = span + 256
    generator = torch.Generator().manual_seed(2931)
    host = (torch.randn(offset + slots * page_stride + 32, generator=generator) * 0.2).to(dtype)
    view = host.as_strided((slots, state_len, dim), (page_stride, row_stride, 1), offset)
    backing = host.npu()
    state = backing.as_strided(view.shape, view.stride(), view.storage_offset())
    # A separately allocated dense tensor is the native control. Never make the
    # control a flat-storage view: its own storage shape must genuinely be 3D.
    dense = view.contiguous().npu()
    return host, backing, state, dense


def _metadata(batch, slots, mode):
    indices = torch.arange(1, batch + 1, dtype=torch.int32)
    lengths = torch.ones(batch, dtype=torch.int32)
    if mode == "null":
        indices.zero_()
    elif mode == "empty":
        lengths.zero_()
    else:
        if mode == "reordered":
            indices = indices.flip(0)
        indices[1:5] = torch.tensor([0, -1, slots, -2], dtype=torch.int32)
        # Fixed-row decode must keep x[request] addressing after this gap.
        lengths[6] = 0
    starts = torch.cat((torch.zeros(1, dtype=torch.int32), lengths.cumsum(0).to(torch.int32)))
    valid = (indices > 0) & (indices < slots) & (lengths > 0)
    normalized = torch.where(valid, indices, 0)
    return indices, starts, normalized, valid


def _attention(*, mtp=0):
    cls = bailing_kda.AscendBailingMoeV3KimiDeltaAttention
    result = cls.__new__(cls)
    torch.nn.Module.__init__(result)
    result.num_speculative_tokens = mtp
    result._conv_state_dim_first = False
    return result


def _run(attention, state, x, weight, indices, starts, *, max_query_len=1, accepted=None, initial=None, run_mode=1):
    return attention._run_causal_conv1d(
        x,
        weight,
        state,
        starts,
        indices,
        initial,
        run_mode=run_mode,
        max_query_len=max_query_len,
        num_accepted_tokens=accepted,
    )


def _spies(monkeypatch):
    copies = MagicMock(wraps=bailing_kda.copy_conv_state)
    fused = MagicMock(wraps=bailing_kda.gather_update_conv_state)
    monkeypatch.setattr(bailing_kda, "copy_conv_state", copies)
    monkeypatch.setattr(bailing_kda, "gather_update_conv_state", fused)
    return copies, fused


def _assert_direct(spies):
    for spy in spies:
        spy.assert_not_called()


def _cpu_decode_update(before, state, x, indices, starts):
    after = before.clone()
    view = after.as_strided(state.shape, state.stride(), state.storage_offset())
    for request, slot in enumerate(indices.tolist()):
        if not (0 < slot < state.shape[0] and starts[request + 1] > starts[request]):
            continue
        old = view[slot].clone()
        view[slot, :2] = old[1:3]
        view[slot, 2] = x[request]
    return after


def _assert_state(backing, state, dense, expected):
    torch.testing.assert_close(backing.cpu(), expected, rtol=0, atol=0)
    expected_view = expected.as_strided(state.shape, state.stride(), state.storage_offset())
    torch.testing.assert_close(dense.cpu(), expected_view, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("layout", ["page", "row_page"])
@pytest.mark.parametrize("state_len", [3, 6])
@torch.inference_mode()
def test_native_pagegap_decode_preserves_full_storage(monkeypatch, dtype, layout, state_len):
    spies = _spies(monkeypatch)
    expected, backing, state, dense = _states(dtype, state_len=state_len, layout=layout)
    attention = _attention()
    generator = torch.Generator().manual_seed(777)
    weight = (torch.randn(4, state.shape[-1], generator=generator) * 0.2).to(dtype).npu()
    for mode in ("active", "reordered", "empty", "null", "active"):
        indices, starts, normalized, valid = _metadata(8, state.shape[0], mode)
        host_x = (torch.randn(8, state.shape[-1], generator=generator) * 0.2).to(dtype)
        expected = _cpu_decode_update(expected, state, host_x, indices, starts)
        x, nstarts = host_x.npu(), starts.npu()
        reference = causal_conv1d_update(
            x,
            dense,
            weight,
            activation="silu",
            conv_state_indices=normalized.npu(),
            query_start_loc=nstarts,
            max_query_len=1,
            null_block_id=0,
            out=torch.zeros_like(x),
        )
        # The production method accepts a metadata matrix and selects its first
        # column; that column is deliberately noncontiguous before normalization.
        index_matrix = torch.stack((indices, torch.zeros_like(indices)), dim=-1).npu()
        actual = _run(attention, state, x, weight, index_matrix, nstarts)
        actual_host = actual.cpu()
        torch.testing.assert_close(actual_host, reference.cpu(), rtol=0, atol=0)
        assert torch.count_nonzero(actual_host[~valid]).item() == 0
        torch.testing.assert_close(x.cpu(), host_x, rtol=0, atol=0)
        _assert_state(backing, state, dense, expected)
    _assert_direct(spies)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_native_pagegap_graph_replay_changes_metadata(monkeypatch, dtype):
    spies = _spies(monkeypatch)
    expected, backing, state, dense = _states(dtype, batch=128, layout="row_page")
    attention = _attention()
    x = torch.zeros(128, state.shape[-1], dtype=dtype, device="npu")
    weight = torch.full((4, state.shape[-1]), 0.125, dtype=dtype, device="npu")
    indices = torch.zeros(128, dtype=torch.int32, device="npu")
    starts = torch.arange(129, dtype=torch.int32, device="npu")
    _run(attention, state, x, weight, indices, starts)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = _run(attention, state, x, weight, indices, starts)
    generator = torch.Generator().manual_seed(238)
    for mode in ("active", "reordered", "empty", "null", "active"):
        host_indices, host_starts, normalized, valid = _metadata(128, state.shape[0], mode)
        host_x = (torch.randn(x.shape, generator=generator) * 0.2).to(dtype)
        expected = _cpu_decode_update(expected, state, host_x, host_indices, host_starts)
        x.copy_(host_x)
        indices.copy_(host_indices)
        starts.copy_(host_starts)
        reference = causal_conv1d_update(
            x,
            dense,
            weight,
            activation="silu",
            conv_state_indices=normalized.npu(),
            query_start_loc=starts,
            max_query_len=1,
            null_block_id=0,
            out=torch.zeros_like(x),
        )
        graph.replay()
        actual_host = actual.cpu()
        torch.testing.assert_close(actual_host, reference.cpu(), rtol=0, atol=0)
        assert torch.count_nonzero(actual_host[~valid]).item() == 0
        _assert_state(backing, state, dense, expected)
    _assert_direct(spies)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_native_pagegap_prefill_initial_state_modes(monkeypatch, dtype):
    spies = _spies(monkeypatch)
    expected, backing, state, dense = _states(dtype, layout="row_page")
    attention = _attention()
    lengths = torch.tensor([1, 3, 0, 5, 2, 1, 4, 2], dtype=torch.int32)
    starts = torch.cat((torch.zeros(1, dtype=torch.int32), lengths.cumsum(0).to(torch.int32))).npu()
    indices = torch.arange(1, 9, dtype=torch.int32, device="npu")
    initial = torch.tensor([True, False, True, False, True, False, True, False], device="npu")
    x = torch.randn(int(lengths.sum()), state.shape[-1], dtype=dtype, device="npu") * 0.2
    weight = torch.full((4, state.shape[-1]), 0.125, dtype=dtype, device="npu")
    reference = causal_conv1d_fn(
        x,
        weight,
        None,
        conv_states=dense,
        query_start_loc=starts,
        cache_indices=indices,
        has_initial_state=initial,
        activation="silu",
        pad_slot_id=-1,
        null_block_id=0,
    )
    actual = _run(attention, state, x, weight, indices, starts, initial=initial, run_mode=0)
    torch.testing.assert_close(actual.cpu(), reference.cpu(), rtol=0, atol=0)
    # Native dense prefill is the history oracle here; preserve physical gaps.
    expected.as_strided(state.shape, state.stride(), state.storage_offset()).copy_(dense.cpu())
    torch.testing.assert_close(backing.cpu(), expected, rtol=0, atol=0)
    _assert_direct(spies)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tokens", [3, 8])
@torch.inference_mode()
def test_native_pagegap_speculative_history(monkeypatch, dtype, tokens):
    spies = _spies(monkeypatch)
    expected, backing, state, dense = _states(dtype, state_len=tokens + 2, layout="row_page")
    attention = _attention(mtp=tokens - 1)
    host_indices = torch.tensor([1, 0, -1, state.shape[0], 5, 6, 7, 8], dtype=torch.int32)
    normalized = torch.where((host_indices > 0) & (host_indices < state.shape[0]), host_indices, 0)
    starts = torch.arange(0, (8 + 1) * tokens, tokens, dtype=torch.int32, device="npu")
    accepted = (torch.arange(8, dtype=torch.int32) % tokens + 1).npu()
    x = torch.randn(8 * tokens, state.shape[-1], dtype=dtype, device="npu") * 0.2
    weight = torch.full((4, state.shape[-1]), 0.125, dtype=dtype, device="npu")
    reference = causal_conv1d_update(
        x,
        dense,
        weight,
        activation="silu",
        conv_state_indices=normalized.npu(),
        num_accepted_tokens=accepted,
        query_start_loc=starts,
        max_query_len=tokens,
        null_block_id=0,
        out=torch.zeros_like(x),
    )
    actual = _run(
        attention,
        state,
        x,
        weight,
        host_indices.npu(),
        starts,
        max_query_len=tokens,
        accepted=accepted,
    )
    actual_host = actual.cpu()
    torch.testing.assert_close(actual_host, reference.cpu(), rtol=0, atol=0)
    assert torch.count_nonzero(actual_host.reshape(8, tokens, -1)[normalized == 0]).item() == 0
    expected.as_strided(state.shape, state.stride(), state.storage_offset()).copy_(dense.cpu())
    torch.testing.assert_close(backing.cpu(), expected, rtol=0, atol=0)
    _assert_direct(spies)


@torch.inference_mode()
def test_native_pagegap_empty_batch_does_not_touch_state(monkeypatch):
    spies = _spies(monkeypatch)
    expected, backing, state, _ = _states(torch.bfloat16)
    x = torch.empty(0, state.shape[-1], dtype=torch.bfloat16, device="npu")
    weight = torch.ones(4, state.shape[-1], dtype=torch.bfloat16, device="npu")
    actual = _run(
        _attention(),
        state,
        x,
        weight,
        torch.empty(0, 1, dtype=torch.int32, device="npu"),
        torch.zeros(1, dtype=torch.int32, device="npu"),
    )
    assert actual.shape == x.shape
    torch.testing.assert_close(backing.cpu(), expected, rtol=0, atol=0)
    _assert_direct(spies)
