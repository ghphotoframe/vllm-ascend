# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Paged MLA reads select a compatible reader without copying the cache pool."""

from unittest.mock import patch

import pytest
import torch
import torch_npu

from vllm_ascend.device.device_op import A5DeviceAdaptor, BaseDeviceAdaptor, DeviceOperator
from vllm_ascend.device.paged_mla_cache_read import try_load_strided_mla_cache

BLOCK_SIZE = 128
KEY_DIM = 512
ROPE_DIM = 64
PREFIX_ELEMENTS = 256
PAGE_PADDING_ELEMENTS = 256


def _cache_views(
    backing: torch.Tensor, blocks: int, layout: str = "component_major", block_size: int = BLOCK_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    page_stride = block_size * (KEY_DIM + ROPE_DIM) + PAGE_PADDING_ELEMENTS
    if layout == "token_fused":
        key_token_stride = rope_token_stride = KEY_DIM + ROPE_DIM
        rope_offset = KEY_DIM
    else:
        key_token_stride, rope_token_stride = KEY_DIM, ROPE_DIM
        rope_offset = block_size * KEY_DIM
    key = torch.as_strided(
        backing,
        (blocks, block_size, 1, KEY_DIM),
        (page_stride, key_token_stride, key_token_stride, 1),
        storage_offset=PREFIX_ELEMENTS,
    )
    rope = torch.as_strided(
        backing,
        (blocks, block_size, 1, ROPE_DIM),
        (page_stride, rope_token_stride, rope_token_stride, 1),
        storage_offset=PREFIX_ELEMENTS + rope_offset,
    )
    return key, rope


def _reference_rows(cache: torch.Tensor, table: torch.Tensor, lengths: list[int], starts: list[int]) -> torch.Tensor:
    # Resolve one token at a time on the CPU, independently of the device
    # gather's block traversal and the old repeat_interleave/index chain.
    rows = [
        cache[int(table[request, position // cache.shape[1]]), position % cache.shape[1]]
        for request, (length, start) in enumerate(zip(lengths, starts))
        for position in range(start, start + length)
    ]
    return torch.stack(rows) if rows else cache.new_empty((0, 1, cache.shape[-1]))


@pytest.mark.parametrize(
    "adaptor,layout",
    [
        (BaseDeviceAdaptor, "dense"),
        (A5DeviceAdaptor, "dense"),
        (BaseDeviceAdaptor, "component_major"),
        (A5DeviceAdaptor, "component_major"),
        (A5DeviceAdaptor, "token_fused"),
    ],
    ids=["base-dense", "a5-dense", "base-component-major", "a5-component-major", "a5-token-fused"],
)
@pytest.mark.parametrize("block_size", [128, 384])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("no_rope", [False, True], ids=["rope", "nope-duplicate"])
@pytest.mark.parametrize("case", ["tail-1", "tail-7", "tail-last", "mixed-empty", "all-empty"])
def test_mla_gather_preserves_pages_and_selects_compatible_reader(adaptor, layout, block_size, dtype, no_rope, case):
    lengths, starts = {
        "tail-1": ([block_size + 1], [block_size]),
        "tail-7": ([block_size + 7], [2 * block_size]),
        "tail-last": ([2 * block_size - 1], [block_size]),
        "mixed-empty": (
            [block_size + 1, 0, 2 * block_size + 7, block_size - 1],
            [block_size, block_size * 9999, 0, 2 * block_size],
        ),
        "all-empty": ([0, 0], [0, block_size * 9999]),
    }[case]
    blocks = 12
    page_stride = block_size * (KEY_DIM + ROPE_DIM) + PAGE_PADDING_ELEMENTS
    # Poison the prefix, page gaps and suffix. Populate on the CPU so cache
    # setup does not itself exercise a noncontiguous NPU write implementation.
    if layout == "dense":
        key_cpu = torch.empty(blocks, block_size, 1, KEY_DIM, dtype=dtype)
        rope_cpu = torch.empty(blocks, block_size, 1, ROPE_DIM, dtype=dtype)
    else:
        original = torch.full((PREFIX_ELEMENTS + blocks * page_stride + PREFIX_ELEMENTS,), 37, dtype=dtype)
        key_cpu, rope_cpu = _cache_views(original, blocks, layout, block_size)
    key_cpu.copy_((torch.arange(key_cpu.numel(), dtype=torch.int32) % 127 - 63).reshape(key_cpu.shape))
    rope_cpu.copy_((torch.arange(rope_cpu.numel(), dtype=torch.int32) % 97 - 48).reshape(rope_cpu.shape))
    if layout == "dense":
        key_cache, rope_cache = key_cpu.to("npu"), rope_cpu.to("npu")
        assert key_cache.is_contiguous() and rope_cache.is_contiguous()
        source_allocations = [(key_cache, key_cpu), (rope_cache, rope_cpu)]
    else:
        backing = original.to("npu")
        key_cache, rope_cache = _cache_views(backing, blocks, layout, block_size)
        assert not key_cache.is_contiguous() and not rope_cache.is_contiguous()
        assert key_cache.storage_offset() > 0 and rope_cache.storage_offset() > 0
        assert key_cache.stride(0) > block_size * KEY_DIM
        assert rope_cache.stride(0) > block_size * ROPE_DIM
        if layout == "token_fused":
            assert key_cache.stride(1) == rope_cache.stride(1) == KEY_DIM + ROPE_DIM
        source_allocations = [(backing, original)]

    # The zero-length request deliberately has unusable entries and a large
    # aligned start: no cache page may be read for that row.
    table_cpu = torch.tensor(
        [[9, 2, 7, 1, 10, 3], [-1, -1, -1, -1, -1, -1], [8, 0, 11, 4, 2, 5], [6, 10, 3, 8, 1, 4]],
        dtype=torch.int32,
    )[: len(lengths)]
    # Slice a wider tensor to exercise the wrapper's existing seq_lens
    # contiguous conversion without changing the per-request length ABI.
    lengths_npu = torch.tensor([[length, 0] for length in lengths], dtype=torch.int32, device="npu")[:, 0]
    starts_npu = torch.tensor(starts, dtype=torch.int32, device="npu")
    value_cache = key_cache if no_rope else rope_cache
    expected_key = _reference_rows(key_cpu, table_cpu, lengths, starts)
    expected_value = expected_key if no_rope else _reference_rows(rope_cpu, table_cpu, lengths, starts)
    key = torch.full(expected_key.shape, -77, dtype=dtype, device="npu")
    value = torch.full(expected_value.shape, -77, dtype=dtype, device="npu")

    # MLA-NoPE passes the latent cache twice, rather than an empty RoPE cache:
    # the native API treats any empty operand as a no-op for both outputs.
    with (
        patch(
            "vllm_ascend.device.device_op.torch_npu.npu_gather_pa_kv_cache",
            wraps=torch_npu.npu_gather_pa_kv_cache,
        ) as native_gather,
        patch("vllm_ascend.device.device_op.try_load_strided_mla_cache", wraps=try_load_strided_mla_cache) as reader,
    ):
        adaptor.kv_cache_load(
            key_cache, value_cache, table_cpu.to("npu"), lengths_npu, starts_npu, key=key, value=value
        )
    reader.assert_called_once()
    if layout == "dense":
        native_gather.assert_called_once()
    else:
        native_gather.assert_not_called()
    assert key.is_contiguous() and value.is_contiguous()
    torch.testing.assert_close(key.cpu(), expected_key, rtol=0, atol=0)
    torch.testing.assert_close(value.cpu(), expected_value, rtol=0, atol=0)
    # Includes untouched pages and all poison regions of the shared allocation.
    for source, original_source in source_allocations:
        torch.testing.assert_close(source.cpu(), original_source, rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [257, 130944], ids=["short-prefix", "profile-prefix"])
@pytest.mark.parametrize("layout", ["dense", "component_major", "token_fused"])
def test_mla_gather_does_not_materialize_the_cache_pool(tokens, layout):
    blocks = 2048
    page_stride = BLOCK_SIZE * (KEY_DIM + ROPE_DIM) + PAGE_PADDING_ELEMENTS
    if layout == "dense":
        key_cache = torch.empty(blocks, BLOCK_SIZE, 1, KEY_DIM, dtype=torch.bfloat16, device="npu")
        rope_cache = torch.empty(blocks, BLOCK_SIZE, 1, ROPE_DIM, dtype=torch.bfloat16, device="npu")
    else:
        backing = torch.empty(PREFIX_ELEMENTS + blocks * page_stride, dtype=torch.bfloat16, device="npu")
        key_cache, rope_cache = _cache_views(backing, blocks, layout)
    key_cache.fill_(3)
    rope_cache.fill_(7)
    table_columns = (tokens + BLOCK_SIZE - 1) // BLOCK_SIZE + 1
    table = torch.arange(blocks - 1, blocks - table_columns - 1, -1, dtype=torch.int32, device="npu").unsqueeze(0)
    lengths = torch.tensor([tokens], dtype=torch.int32, device="npu")
    starts = torch.tensor([BLOCK_SIZE], dtype=torch.int32, device="npu")
    key = torch.empty((tokens, 1, KEY_DIM), dtype=torch.bfloat16, device="npu")
    rope = torch.empty((tokens, 1, ROPE_DIM), dtype=torch.bfloat16, device="npu")

    # Initialize the operator and its workspace before tracking transient live
    # memory. Outputs are already allocated. The fallback additionally needs
    # selected-row packed storage and O(tokens) indices; it must not copy the
    # full 256 MiB latent pool. Native retains the original 128 MiB allowance.
    # This cannot detect copies inside externally managed or preallocated
    # workspace; a kernel trace must check those separately.
    DeviceOperator.kv_cache_load(key_cache, rope_cache, table, lengths, starts, key=key, value=rope)
    torch.npu.synchronize()
    baseline = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    with patch(
        "vllm_ascend.device.device_op.torch_npu.npu_gather_pa_kv_cache",
        wraps=torch_npu.npu_gather_pa_kv_cache,
    ) as native_gather:
        DeviceOperator.kv_cache_load(key_cache, rope_cache, table, lengths, starts, key=key, value=rope)
        torch.npu.synchronize()
    peak_bytes = torch.npu.max_memory_allocated() - baseline
    full_latent_pool_bytes = key_cache.numel() * key_cache.element_size()
    if layout == "dense":
        native_gather.assert_called_once()
        limit_bytes = full_latent_pool_bytes // 2
    else:
        native_gather.assert_not_called()
        padded_tokens = (tokens + 7) // 8 * 8
        packed_bytes = padded_tokens * (KEY_DIM + ROPE_DIM) * key_cache.element_size()
        # Eight INT64 arrays conservatively cover request IDs, positions,
        # physical blocks, slots and indexing temporaries. Allow 32 MiB for
        # allocator rounding/workspace. Even at 130944 selected rows this is
        # < 185 MiB, so one full latent-pool copy still exceeds the limit.
        index_bytes = padded_tokens * 8 * 8
        limit_bytes = packed_bytes + index_bytes + 32 * 1024**2
        assert limit_bytes < full_latent_pool_bytes
    assert peak_bytes < limit_bytes, f"{layout} MLA gather allocated {peak_bytes} bytes (limit: {limit_bytes})"
    torch.testing.assert_close(key.cpu(), torch.full(key.shape, 3, dtype=key.dtype), rtol=0, atol=0)
    torch.testing.assert_close(rope.cpu(), torch.full(rope.shape, 7, dtype=rope.dtype), rtol=0, atol=0)
