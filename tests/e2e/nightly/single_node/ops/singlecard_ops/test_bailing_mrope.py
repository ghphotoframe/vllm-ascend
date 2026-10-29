# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real NPU rotary and replay checks; no model weights or distributed launch."""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config

import vllm_ascend.ops  # noqa: F401  # Register ops before attention/device imports.
from vllm_ascend.attention.mla_v1 import AscendMLAImpl, AscendMLAMetadataBuilder
from vllm_ascend.ops.bailing_rotary_embedding import AscendBailingMRotaryEmbedding
from vllm_ascend.ops.rotary_embedding import AscendApplyRotaryEmb
from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import AscendAutoRegressiveSpeculator


@pytest.fixture(autouse=True)
def npu_config():
    torch.npu.set_device(0)
    with set_current_vllm_config(VllmConfig()):
        yield


def _embedding(dtype):
    embedding = AscendBailingMRotaryEmbedding(80, 64, 256, 6000000, False, dtype, [8, 12, 12]).to("npu")
    embedding.apply_rotary_emb = AscendApplyRotaryEmb(enforce_enable=True, is_neox_style=False)
    return embedding


def _reference_coefficients(positions, dtype):
    positions = positions.cpu()
    if positions.ndim == 1:
        positions = positions.expand(3, -1)
    inv_freq = 1 / 6000000 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64)
    axes = torch.tensor([1, 2] * 12 + [0] * 8)
    selected_positions = positions[axes].T
    angles = selected_positions.float() * inv_freq
    return angles.cos().to(dtype), angles.sin().to(dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("three_axes", [False, True])
@torch.inference_mode()
def test_bailing_rotary_negative_axes_strided_views_and_mla(dtype, three_axes):
    embedding = _embedding(dtype)
    positions_backing = torch.tensor([[7, 0, 8, 0, 9, 0], [-2, 0, 0, 0, 2, 0], [3, 0, -1, 0, -3, 0]], device="npu")
    positions = positions_backing[:, ::2] if three_axes else positions_backing[0, ::2]
    query_backing = torch.randn(3, 4, 96, dtype=dtype, device="npu")
    key_backing = torch.randn(3, 1, 96, dtype=dtype, device="npu")
    query, key = query_backing[..., :80], key_backing[..., :80]
    original_query, original_key = query_backing.clone(), key_backing.clone()
    cos_cpu, sin_cpu = _reference_coefficients(positions, dtype)

    def reference(x):
        x = x.cpu()
        x1, x2 = x[..., :64:2], x[..., 1:64:2]
        rotated = torch.stack(
            (x1 * cos_cpu[:, None] - x2 * sin_cpu[:, None], x2 * cos_cpu[:, None] + x1 * sin_cpu[:, None]), -1
        ).flatten(-2)
        return torch.cat((rotated, x[..., 64:]), -1)

    tolerance = 0.02 if dtype == torch.bfloat16 else 0.002
    actual_q, actual_k = embedding.forward_oot(positions, query, key)
    torch.testing.assert_close(actual_q.cpu(), reference(query), rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(actual_k.cpu(), reference(key), rtol=tolerance, atol=tolerance)
    cos, sin = embedding.select_cos_sin_mla(positions)
    impl = AscendMLAImpl.__new__(AscendMLAImpl)
    impl.use_mla_rope = True
    actual_mla = impl.rope_single(query[..., :64].contiguous(), cos, sin)
    reference_rope = reference(query)[..., :64]
    # MLA's npu_interleave_rope also changes GPT-J pairs to even/odd halves.
    expected_mla = torch.cat((reference_rope[..., 0::2], reference_rope[..., 1::2]), dim=-1)
    torch.testing.assert_close(actual_mla.cpu(), expected_mla, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(actual_q[..., 64:], query[..., 64:], rtol=0, atol=0)
    torch.testing.assert_close(query_backing, original_query, rtol=0, atol=0)
    torch.testing.assert_close(key_backing, original_key, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_bailing_mla_graph_reloads_three_axes_and_mtp_delta(dtype):
    embedding = _embedding(dtype)
    builder = AscendMLAMetadataBuilder.__new__(AscendMLAMetadataBuilder)
    builder.bailing_rotary_emb = embedding
    builder.cos_cache = torch.empty((4, 1, 1, 64), dtype=dtype, device="npu")
    builder.sin_cache = torch.empty_like(builder.cos_cache)
    positions = torch.tensor([[7, 8, 9, 0], [-2, 0, 2, 0], [3, -1, -3, 0]], device="npu")
    pointers = (builder.cos_cache.data_ptr(), builder.sin_cache.data_ptr())

    builder._get_cos_and_sin(positions, use_cache=True)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        cos, sin = builder._get_cos_and_sin(positions, use_cache=True)
    for reordered in (positions.clone().neg(), positions[:, [2, 0, 1, 3]].clone(), torch.zeros_like(positions)):
        positions.copy_(reordered)
        graph.replay()
        expected_cos, expected_sin = _reference_coefficients(reordered, dtype)
        torch.testing.assert_close(cos[:, 0, 0].cpu(), expected_cos.repeat(1, 2), rtol=0, atol=0)
        torch.testing.assert_close(sin[:, 0, 0].cpu(), expected_sin.repeat(1, 2), rtol=0, atol=0)
        assert (cos.data_ptr(), sin.data_ptr()) == pointers

    speculator = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    speculator._mrope_decode_delta = torch.zeros(4, dtype=torch.long, device="npu")
    speculator.input_buffers = SimpleNamespace(positions=torch.tensor([40, 60, 0, 0], device="npu"))
    speculator.model_state = SimpleNamespace(rope_state=SimpleNamespace(prefill_delta=SimpleNamespace(gpu=None)))
    absolute = speculator.input_buffers.positions.clone()
    delta_pointer = speculator._mrope_decode_delta.data_ptr()
    batch = SimpleNamespace(num_reqs=2, idx_mapping=torch.tensor([2, 0], device="npu"))
    speculator.model_state.rope_state.prefill_delta.gpu = torch.tensor([3, 0, -7], device="npu")
    speculator._prepare_mrope_decode_delta(batch, dummy_run=False)
    assert speculator._get_mrope_decode_positions(4, 0) is None
    builder._get_cos_and_sin(speculator._get_mrope_decode_positions(4, 1), use_cache=True)
    torch.npu.synchronize()
    draft_graph = torch.npu.NPUGraph()
    with torch.npu.graph(draft_graph):
        draft_positions = speculator._get_mrope_decode_positions(4, 1)
        draft_cos, draft_sin = builder._get_cos_and_sin(draft_positions, use_cache=True)
    # Replace the source allocation, as UVA staged writes do between batches.
    for source, order in [([9, 4, -5], [2, 0]), ([-8, 6, 2], [1, 2])]:
        speculator.model_state.rope_state.prefill_delta.gpu = torch.tensor(source, device="npu")
        batch.idx_mapping.copy_(torch.tensor(order, device="npu"))
        speculator._prepare_mrope_decode_delta(batch, dummy_run=False)
        draft_graph.replay()
        expected = absolute.cpu() + torch.tensor([source[i] for i in order] + [0, 0])
        expected_cos, expected_sin = _reference_coefficients(expected, dtype)
        torch.testing.assert_close(draft_positions.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(draft_cos[:, 0, 0].cpu(), expected_cos.repeat(1, 2), rtol=0, atol=0)
        torch.testing.assert_close(draft_sin[:, 0, 0].cpu(), expected_sin.repeat(1, 2), rtol=0, atol=0)
        assert speculator._mrope_decode_delta.data_ptr() == delta_pointer
        torch.testing.assert_close(speculator.input_buffers.positions, absolute, rtol=0, atol=0)
