# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real FLA equivalence for fused selected-history gather/update.

No new convolution math: compare against the current gather -> FLA -> scatter
pipeline, including FP32 history's BF16 round trip and untouched physical gaps.
Valid active slots are unique; repeated NULL/PAD slots are explicitly covered.
"""

import argparse
import json
import statistics
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch_npu  # noqa: F401
from fla_npu.ops.ascendc import causal_conv1d_update

from vllm_ascend.ops import bailing_moe_v3_kda as bailing_kda
from vllm_ascend.ops.triton.kda.conv_state import copy_conv_state, gather_update_conv_state


def _state(batch, state_len, dim, layout, dtype):
    slots, offset = batch + 5, 17
    span = state_len * dim
    page = 110592 if layout == "page" else span + 64
    stride = (page, 1, state_len) if layout == "ds" else (page, dim, 1)
    generator = torch.Generator().manual_seed(401)
    # FP32 values intentionally do not round-trip through BF16 unchanged.
    host = torch.randn((slots - 1) * page + offset + span + 31, generator=generator).to(dtype)
    backing = host.npu()
    return backing, backing.as_strided((slots, state_len, dim), stride, offset)


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
        if batch >= 8:
            indices[1:5] = torch.tensor([0, -1, slots, 0])
            # Keep a mixed empty segment: fixed-row decode still reads
            # x[request] after this gap, not the compacted starts[request].
            lengths[6] = 0
    starts = torch.cat((torch.zeros(1, dtype=torch.int32), lengths.cumsum(0).to(torch.int32)))
    return indices, starts


def _run(cache, x, weight, indices, starts, *, fused):
    if fused == "integrated":
        cls = bailing_kda.AscendBailingMoeV3KimiDeltaAttention
        attention = cls.__new__(cls)
        torch.nn.Module.__init__(attention)
        attention.num_speculative_tokens = 0
        attention._conv_state_dim_first = cache.stride(1) == 1
        storage = cache.transpose(-1, -2) if attention._conv_state_dim_first else cache
        return attention._run_causal_conv1d(
            x,
            weight,
            storage,
            starts,
            indices.unsqueeze(-1),
            None,
            run_mode=1,
            max_query_len=1,
            num_accepted_tokens=None,
        )
    requests = indices.shape[0]
    packed = torch.empty((requests, *cache.shape[1:]), dtype=torch.bfloat16, device=cache.device)
    packed_indices = torch.empty(requests, dtype=torch.int32, device=cache.device)
    if fused:
        gather_update_conv_state(cache, packed, x, indices, starts, packed_indices)
    else:
        selected = torch.where(indices == 0, -1, indices)
        copy_conv_state(cache, packed, selected, starts, packed_indices, write_back=False)
        packed_indices = torch.where(packed_indices == -1, requests, packed_indices)
    output = causal_conv1d_update(
        x,
        packed,
        weight,
        activation="silu",
        conv_state_indices=packed_indices,
        query_start_loc=starts,
        max_query_len=1,
        null_block_id=requests,
        out=torch.zeros_like(x),
    )
    if not fused:
        copy_conv_state(cache, packed, selected, starts, packed_indices, write_back=True)
    return output


def _expected_update(backing, cache, x, indices, starts):
    result = backing.clone()
    state = result.as_strided(cache.shape, cache.stride(), cache.storage_offset())
    for request, slot in enumerate(indices.tolist()):
        start, end = starts[request : request + 2].tolist()
        if not (0 < slot < cache.shape[0] and end - start == 1):
            continue
        rounded = state[slot].to(torch.bfloat16).clone()
        rounded[0] = rounded[1]
        rounded[1] = rounded[2]
        # Exact row count selects FLA's fixed-row decode mode; a different
        # row count selects compact varlen mode in the existing fallback.
        rounded[2] = x[request if x.shape[0] == indices.numel() else start]
        state[slot] = rounded.to(state.dtype)
    return result


@pytest.mark.parametrize("implementation", ["helper", "integrated"])
@pytest.mark.parametrize("layout", ["sd", "ds", "page"])
@pytest.mark.parametrize("state_len", [3, 6])
@pytest.mark.parametrize("batch", [1, 128])
@torch.inference_mode()
def test_fused_history_matches_real_fla_step_by_step(monkeypatch, implementation, layout, state_len, batch):
    spy = MagicMock(wraps=gather_update_conv_state)
    monkeypatch.setattr(bailing_kda, "gather_update_conv_state", spy)
    dim = 1536
    left_backing, left = _state(batch, state_len, dim, layout, torch.float32)
    right_backing = left_backing.clone()
    right = right_backing.as_strided(left.shape, left.stride(), left.storage_offset())
    generator = torch.Generator().manual_seed(817)
    weight = (torch.randn(4, dim, generator=generator) * 0.2).bfloat16().npu()
    expected = left_backing.cpu()
    for mode in ("null", "active", "reordered", "empty", "active"):
        indices, starts = _metadata(batch, left.shape[0], mode)
        x = (torch.randn(batch, dim, generator=generator) * 0.2).bfloat16()
        expected = _expected_update(expected, left, x, indices, starts)
        nx, ni, ns = x.npu(), indices.npu(), starts.npu()
        baseline = _run(left, nx, weight, ni, ns, fused=False)
        candidate = _run(right, nx, weight, ni, ns, fused=implementation)
        torch.testing.assert_close(candidate.cpu(), baseline.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(left_backing.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(right_backing.cpu(), expected, rtol=0, atol=0)
    assert spy.call_count == (5 if implementation == "integrated" else 0)


@pytest.mark.parametrize("layout", ["ds", "page"])
def test_maximum_history_fits_ub_and_preserves_state(monkeypatch, layout):
    """Exercise the longest supported history in the actual integrated path."""
    test_fused_history_matches_real_fla_step_by_step(monkeypatch, "integrated", layout, 32, 128)


@pytest.mark.parametrize("implementation", ["helper", "integrated"])
@pytest.mark.parametrize("cache_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layout", ["page", "ds"])
@torch.inference_mode()
def test_graph_replay_changes_indices_and_preserves_extra_history(monkeypatch, implementation, cache_dtype, layout):
    spy = MagicMock(wraps=gather_update_conv_state)
    monkeypatch.setattr(bailing_kda, "gather_update_conv_state", spy)
    batch, dim, state_len = 128, 1536, 6
    left_backing, left = _state(batch, state_len, dim, layout, cache_dtype)
    right_backing = left_backing.clone()
    right = right_backing.as_strided(left.shape, left.stride(), left.storage_offset())
    x = torch.empty(batch, dim, dtype=torch.bfloat16, device="npu")
    weight = torch.full((4, dim), 0.25, dtype=torch.bfloat16, device="npu")
    indices = torch.zeros(batch, dtype=torch.int32, device="npu")
    starts = torch.arange(batch + 1, dtype=torch.int32, device="npu")
    x.zero_()
    _run(left, x, weight, indices, starts, fused=False)
    _run(right, x, weight, indices, starts, fused=implementation)
    torch.npu.synchronize()
    before = left_backing.cpu()
    graphs, outputs = [], []
    for state, fused in ((left, False), (right, implementation)):
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = _run(state, x, weight, indices, starts, fused=fused)
        graphs.append(graph)
        outputs.append(output)
    generator = torch.Generator().manual_seed(991)
    for mode in ("active", "reordered", "empty", "null"):
        host_indices, host_starts = _metadata(batch, left.shape[0], mode)
        host_x = (torch.randn(batch, dim, generator=generator) * 0.2).bfloat16()
        before = _expected_update(before, left, host_x, host_indices, host_starts)
        x.copy_(host_x)
        indices.copy_(host_indices)
        starts.copy_(host_starts)
        torch.npu.synchronize()
        for graph in graphs:
            graph.replay()
        torch.testing.assert_close(outputs[0].cpu(), outputs[1].cpu(), rtol=0, atol=0)
        torch.testing.assert_close(left_backing.cpu(), before, rtol=0, atol=0)
        torch.testing.assert_close(right_backing.cpu(), before, rtol=0, atol=0)
    staged = cache_dtype == torch.float32 or layout == "ds"
    assert spy.call_count == (2 if implementation == "integrated" and staged else 0)


@torch.inference_mode()
def test_selected_memory_does_not_scale_with_pool_rows():
    for slots in (8, 2048):
        backing = torch.randn(slots * (3 * 1536 + 64) + 17, dtype=torch.float32, device="npu")
        state = backing.as_strided((slots, 3, 1536), (3 * 1536 + 64, 1536, 1), 17)
        packed = torch.empty(1, 3, 1536, dtype=torch.bfloat16, device="npu")
        x = torch.full((1, 1536), 0.25, dtype=torch.bfloat16, device="npu")
        indices = torch.tensor([slots - 1], dtype=torch.int32, device="npu")
        starts = torch.tensor([0, 1], dtype=torch.int32, device="npu")
        packed_indices = torch.empty_like(indices)
        gather_update_conv_state(state, packed, x, indices, starts, packed_indices)
        torch.npu.synchronize()
        before = torch.npu.memory_allocated()
        torch.npu.reset_peak_memory_stats()
        gather_update_conv_state(state, packed, x, indices, starts, packed_indices)
        torch.npu.synchronize()
        assert torch.npu.max_memory_allocated() - before <= 1024 * 1024


@torch.inference_mode()
def test_sliced_channels_use_integrated_fallback(monkeypatch):
    """Existing copy supports channel holes that the fused helper rejects."""
    batch, dim, state_len = 128, 1536, 6
    left_backing, wide_state = _state(batch, state_len, dim * 2, "sd", torch.float32)
    left = wide_state[:, :, ::2]
    right_backing = left_backing.clone()
    right = right_backing.as_strided(left.shape, left.stride(), left.storage_offset())
    indices, starts = _metadata(batch, left.shape[0], "active")
    x = torch.randn(batch, dim, dtype=torch.bfloat16)
    weight = torch.randn(4, dim, dtype=torch.bfloat16, device="npu")
    expected = _expected_update(left_backing.cpu(), left, x, indices, starts)
    nx, ni, ns = x.npu(), indices.npu(), starts.npu()
    packed = torch.empty(batch, state_len, dim, dtype=torch.bfloat16, device="npu")
    packed_indices = torch.empty_like(ni)
    with pytest.raises(ValueError, match="history/channel axes"):
        gather_update_conv_state(right, packed, nx, ni, ns, packed_indices)
    fast = MagicMock(wraps=gather_update_conv_state)
    copies = MagicMock(wraps=copy_conv_state)
    monkeypatch.setattr(bailing_kda, "gather_update_conv_state", fast)
    monkeypatch.setattr(bailing_kda, "copy_conv_state", copies)
    baseline = _run(left, nx, weight, ni, ns, fused=False)
    candidate = _run(right, nx, weight, ni, ns, fused="integrated")
    fast.assert_not_called()
    assert copies.call_count == 2
    torch.testing.assert_close(candidate.cpu(), baseline.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(left_backing.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(right_backing.cpu(), expected, rtol=0, atol=0)


@torch.inference_mode()
def test_compact_varlen_input_keeps_existing_fallback(monkeypatch):
    """FLA changes addressing when input rows differ from the request count."""
    batch, dim, state_len = 128, 1536, 6
    left_backing, left = _state(batch, state_len, dim, "page", torch.float32)
    right_backing = left_backing.clone()
    right = right_backing.as_strided(left.shape, left.stride(), left.storage_offset())
    indices, starts = _metadata(batch, left.shape[0], "active")
    x = torch.randn(int(starts[-1]), dim, dtype=torch.bfloat16)
    assert x.shape[0] == batch - 1
    weight = torch.randn(4, dim, dtype=torch.bfloat16, device="npu")
    expected = _expected_update(left_backing.cpu(), left, x, indices, starts)
    nx, ni, ns = x.npu(), indices.npu(), starts.npu()
    packed = torch.empty(batch, state_len, dim, dtype=torch.bfloat16, device="npu")
    packed_indices = torch.empty_like(ni)
    with pytest.raises(ValueError, match="exactly one x row"):
        gather_update_conv_state(right, packed, nx, ni, ns, packed_indices)
    fast = MagicMock(wraps=gather_update_conv_state)
    copies = MagicMock(wraps=copy_conv_state)
    monkeypatch.setattr(bailing_kda, "gather_update_conv_state", fast)
    monkeypatch.setattr(bailing_kda, "copy_conv_state", copies)
    baseline = _run(left, nx, weight, ni, ns, fused=False)
    candidate = _run(right, nx, weight, ni, ns, fused="integrated")
    fast.assert_not_called()
    assert copies.call_count == 2
    torch.testing.assert_close(candidate.cpu(), baseline.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(left_backing.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(right_backing.cpu(), expected, rtol=0, atol=0)


@torch.inference_mode()
def benchmark(output):
    """Same-runtime event timings after equality tests; no model throughput claim."""
    report = {"scope": "One layer: old gather/FLA/scatter versus fused gather+update/FLA", "results": []}
    for batch in (1, 128):
        dim = 1536
        backing, state = _state(batch, 3, dim, "page", torch.float32)
        other_backing = backing.clone()
        other = other_backing.as_strided(state.shape, state.stride(), state.storage_offset())
        x = torch.full((batch, dim), 0.25, dtype=torch.bfloat16, device="npu")
        weight = torch.full((4, dim), 0.125, dtype=torch.bfloat16, device="npu")
        indices, starts = _metadata(batch, state.shape[0], "active")
        indices, starts = indices.npu(), starts.npu()
        graphs, outputs = {}, {}
        for name, cache, fused in (("baseline", state, False), ("candidate", other, "integrated")):
            for _ in range(3):
                _run(cache, x, weight, indices, starts, fused=fused)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                result = _run(cache, x, weight, indices, starts, fused=fused)
            graphs[name], outputs[name] = graph, result
        for _ in range(20):
            for graph in graphs.values():
                graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(backing, other_backing, rtol=0, atol=0)
        torch.testing.assert_close(outputs["baseline"], outputs["candidate"], rtol=0, atol=0)
        samples = {name: [] for name in graphs}
        for group in range(12):
            order = ("baseline", "candidate") if group % 2 == 0 else ("candidate", "baseline")
            for name in order:
                start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                start.record()
                for _ in range(100):
                    graphs[name].replay()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) * 1000 / 100)
        report["results"].append(
            {
                "batch": batch,
                "samples_us": samples,
                "summary_us": {
                    name: {"mean": statistics.mean(values), "median": statistics.median(values)}
                    for name, values in samples.items()
                },
            }
        )
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-json", type=Path, required=True)
    args = parser.parse_args()
    if args.benchmark_json.exists():
        raise SystemExit("Refusing to overwrite benchmark evidence")
    benchmark(args.benchmark_json)
