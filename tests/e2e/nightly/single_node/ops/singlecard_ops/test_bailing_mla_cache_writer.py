# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual NPU bounded KV writer for Ling VL's direct-Q hybrid cache."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu

import vllm_ascend.ops  # noqa: F401  # Register ops before attention/device imports.
from vllm_ascend.attention.mla_v1 import AscendMLAImpl
from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily, get_current_hardware_profile

BLOCK_SIZE = 256
NOPE_DIM = 512
ROPE_DIM = 64
FUSED_DIM = NOPE_DIM + ROPE_DIM
SENTINEL = -37


def _impl():
    torch.npu.set_device(0)
    impl = AscendMLAImpl.__new__(AscendMLAImpl)
    impl.rotary_emb = SimpleNamespace(uses_mla_mrope=True)
    impl.q_lora_rank = impl.fused_qkv_a_proj = impl.q_a_layernorm = None
    impl.support_fp8_attention = impl.is_draft_model = False
    impl.use_mla_rope = True
    impl.dtype = torch.bfloat16
    impl.kv_lora_rank, impl.qk_rope_head_dim, impl.num_kv_heads = NOPE_DIM, ROPE_DIM, 1
    impl.fa_quant_layer = impl.enable_kv_nz = impl.pcp_enabled = impl.dcp_enabled = False
    impl.kv_a_layernorm = SimpleNamespace(
        weight=torch.rand(NOPE_DIM, dtype=impl.dtype, device="npu"), variance_epsilon=1e-6
    )
    impl.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[4, 16]),
        cache_config=SimpleNamespace(block_size=BLOCK_SIZE),
    )
    impl._strided_kv_workspace = None
    impl._prepare_strided_kv_workspace()
    assert impl._uses_strided_kv_adapter()
    return impl


def _cache(layout, blocks=3, offset=16):
    page = BLOCK_SIZE * FUSED_DIM + 128
    backing = torch.full((offset + blocks * page + 32,), SENTINEL, dtype=torch.bfloat16, device="npu")
    if layout == "component_major":
        nope = backing.as_strided((blocks, BLOCK_SIZE, 1, NOPE_DIM), (page, NOPE_DIM, NOPE_DIM, 1), offset)
        rope = backing.as_strided(
            (blocks, BLOCK_SIZE, 1, ROPE_DIM), (page, ROPE_DIM, ROPE_DIM, 1), offset + BLOCK_SIZE * NOPE_DIM
        )
    else:
        nope = backing.as_strided((blocks, BLOCK_SIZE, 1, NOPE_DIM), (page, FUSED_DIM, FUSED_DIM, 1), offset)
        rope = backing.as_strided((blocks, BLOCK_SIZE, 1, ROPE_DIM), (page, FUSED_DIM, FUSED_DIM, 1), offset + NOPE_DIM)
    return backing, (nope, rope)


def _inputs():
    kv = torch.randn(4, FUSED_DIM, dtype=torch.bfloat16, device="npu")
    angles = torch.randn(4, ROPE_DIM // 2, device="npu")
    cos = angles.cos().to(kv.dtype).repeat(1, 2)[:, None, None]
    sin = angles.sin().to(kv.dtype).repeat(1, 2)[:, None, None]
    slots = torch.tensor([BLOCK_SIZE - 1, BLOCK_SIZE, 2 * BLOCK_SIZE + 5, -1], device="npu")
    return kv, cos, sin, slots


def _dense_reference(impl, kv, cos, sin):
    nope = torch.empty(1, BLOCK_SIZE, 1, NOPE_DIM, dtype=kv.dtype, device=kv.device)
    rope = torch.empty(1, BLOCK_SIZE, 1, ROPE_DIM, dtype=kv.dtype, device=kv.device)
    torch_npu.npu_kv_rmsnorm_rope_cache(
        kv.view(4, 1, 1, FUSED_DIM),
        impl.kv_a_layernorm.weight,
        cos,
        sin,
        torch.arange(4, dtype=torch.int64, device=kv.device),
        rope,
        nope,
        epsilon=impl.kv_a_layernorm.variance_epsilon,
        cache_mode="PA",
        is_output_kv=False,
    )
    return rope.flatten(0, 1)[:4].cpu(), nope.flatten(0, 1)[:4].cpu()


def _expected_backing(backing, cache, slots, reference):
    expected = backing.cpu().clone()
    for component, values in zip(cache, reversed(reference)):
        view = expected.as_strided(component.shape, component.stride(), component.storage_offset())
        for token, slot in enumerate(slots.cpu().tolist()):
            if slot >= 0:
                view[slot // BLOCK_SIZE, slot % BLOCK_SIZE].copy_(values[token])
    return expected


@pytest.mark.parametrize("layout", ["component_major", "token_fused"])
@pytest.mark.parametrize("offset", [0, 16])
@pytest.mark.parametrize("prefill", [False, True])
@torch.inference_mode()
def test_direct_q_writer_preserves_padding_and_original_storage(layout, offset, prefill):
    impl = _impl()
    backing, cache = _cache(layout, offset=offset)
    kv, cos, sin, slots = _inputs()
    if (
        layout == "token_fused"
        and get_current_hardware_profile().device_adaptor_family != DeviceAdaptorFamily.FP8_OPTIMIZED
    ):
        before = backing.cpu().clone()
        with pytest.raises(RuntimeError, match="requires component-major caches"):
            if prefill:
                impl.exec_kv_prefill(kv, cos, sin, cache, slots)
            else:
                impl.exec_kv_decode(kv, cos, sin, cache, slots, return_current_kv=True)
        torch.testing.assert_close(backing.cpu(), before, rtol=0, atol=0)
        return
    reference = _dense_reference(impl, kv, cos, sin)
    expected = _expected_backing(backing, cache, slots, reference)
    workspace = impl._strided_kv_workspace
    pointers = tuple(t.data_ptr() for t in workspace)
    if prefill:
        rope, nope = impl.exec_kv_prefill(kv, cos, sin, cache, slots)
    else:
        history_rope, history_nope, rope, nope = impl.exec_kv_decode(kv, cos, sin, cache, slots, return_current_kv=True)
        assert history_rope is cache[1] and history_nope is cache[0]
    torch.testing.assert_close(rope.cpu(), reference[0], rtol=0, atol=0)
    torch.testing.assert_close(nope.cpu(), reference[1], rtol=0, atol=0)
    torch.testing.assert_close(backing.cpu(), expected, rtol=0, atol=0)
    assert tuple(t.data_ptr() for t in impl._strided_kv_workspace) == pointers


@pytest.mark.parametrize("layout", ["component_major", "token_fused"])
@torch.inference_mode()
def test_direct_q_writer_graph_replay_updates_slots_and_skips_all_padding(layout):
    impl = _impl()
    backing, cache = _cache(layout)
    kv, cos, sin, slots = _inputs()
    if (
        layout == "token_fused"
        and get_current_hardware_profile().device_adaptor_family != DeviceAdaptorFamily.FP8_OPTIMIZED
    ):
        before = backing.cpu().clone()
        # Graph warmup must reject unsupported history strides before capture.
        with pytest.raises(RuntimeError, match="requires component-major caches"):
            impl.exec_kv_decode(kv, cos, sin, cache, slots)
        torch.testing.assert_close(backing.cpu(), before, rtol=0, atol=0)
        return
    impl.exec_kv_decode(kv, cos, sin, cache, slots)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        impl.exec_kv_decode(kv, cos, sin, cache, slots)
    for current_slots in ([1, BLOCK_SIZE + 2, -1, 2 * BLOCK_SIZE], [-1, -1, -1, -1]):
        backing.fill_(SENTINEL)
        kv.normal_()
        slots.copy_(torch.tensor(current_slots, device="npu"))
        reference = _dense_reference(impl, kv, cos, sin)
        expected = _expected_backing(backing, cache, slots, reference)
        graph.replay()
        torch.testing.assert_close(backing.cpu(), expected, rtol=0, atol=0)


@torch.inference_mode()
def test_direct_q_writer_temporary_memory_does_not_scale_with_history_pool():
    impl = _impl()
    kv, cos, sin, slots = _inputs()
    peaks = []
    for blocks in (128, 2048):
        backing, cache = _cache("component_major", blocks=blocks)
        impl.exec_kv_decode(kv, cos, sin, cache, slots)
        torch.npu.synchronize()
        baseline = torch.npu.memory_allocated()
        torch.npu.reset_peak_memory_stats()
        impl.exec_kv_decode(kv, cos, sin, cache, slots)
        torch.npu.synchronize()
        peaks.append(torch.npu.max_memory_allocated() - baseline)
        del cache, backing
    # 4 current rows use a preallocated 256-row workspace. Even the smaller
    # history pool is >36 MiB. Measured scatter scratch is about 16 MiB;
    # this bound allows that fixed scratch but rejects whole-pool copies.
    assert max(peaks) < 32 * 1024**2, peaks
    assert peaks[1] <= peaks[0] + 4 * 1024**2, peaks
