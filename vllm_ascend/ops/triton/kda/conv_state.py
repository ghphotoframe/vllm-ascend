# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pack and restore strided convolution states for KDA."""

import torch
from vllm.triton_utils import tl, triton

_CONV_STATE_COPY_BLOCK_SIZE = 256
_CONV_STATE_COPY_MAX_PROGRAMS = 65535


@triton.jit
def _gather_update_conv_state_kernel(
    cache,
    packed,
    x,
    cache_indices,
    starts,
    packed_indices,
    cache_stride,
    index_stride,
    start_stride,
    x_stride,
    num_slots,
    num_tokens,
    REQUESTS: tl.constexpr,
    STATE_LEN: tl.constexpr,
    DIM: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
    DIM_STRIDE: tl.constexpr,
    NULL_SLOT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    channel_tiles = tl.cdiv(DIM, BLOCK)
    # Each program owns every history row for its channels, so shifting rows
    # cannot race another channel tile. Distinct active requests must own
    # distinct writable cache slots, as required by the native update kernel.
    for tile in range(tl.program_id(0), REQUESTS * channel_tiles, tl.num_programs(0)):
        request = tile // channel_tiles
        channels = tile % channel_tiles * BLOCK + tl.arange(0, BLOCK)
        slot = tl.load(cache_indices + request * index_stride).to(tl.int64)
        start = tl.load(starts + request * start_stride)
        end = tl.load(starts + (request + 1) * start_stride)
        active = (slot >= 0) & (slot < num_slots) & (slot != NULL_SLOT)
        active = active & (end - start == 1) & (request < num_tokens)
        safe_slot = tl.where(active, slot, 0)
        offsets = channels * DIM_STRIDE
        in_range = channels < DIM
        active_channels = active & in_range
        cache_row = cache + safe_slot * cache_stride
        packed_row = packed + request * STATE_LEN * DIM
        # Snapshot the three rows before shifting any of them. Keep each row
        # as a 1D channel vector: a 2D tl.gather lowers to scalar loops on A3.
        old0 = tl.load(cache_row + offsets, mask=active_channels, other=0).to(packed.dtype.element_ty)
        old1 = tl.load(cache_row + STATE_STRIDE + offsets, mask=active_channels, other=0).to(packed.dtype.element_ty)
        old2 = tl.load(cache_row + 2 * STATE_STRIDE + offsets, mask=active_channels, other=0).to(
            packed.dtype.element_ty
        )
        tl.store(packed_row + channels, old0, mask=in_range)
        tl.store(packed_row + DIM + channels, old1, mask=in_range)
        tl.store(packed_row + 2 * DIM + channels, old2, mask=in_range)
        # FLA's 2D decode mode uses one input row per request, even when
        # query_start_loc contains empty/padded segments before this request.
        current = tl.load(x + request * x_stride + channels, mask=active_channels, other=0)
        tl.store(cache_row + offsets, old1, mask=active_channels)
        tl.store(cache_row + STATE_STRIDE + offsets, old2, mask=active_channels)
        tl.store(cache_row + 2 * STATE_STRIDE + offsets, current, mask=active_channels)
        # Native width-four update leaves the remaining rows in place. The
        # old gather/scatter still rounds them through BF16. Rows and channels
        # do not overlap, so each tail row can be loaded and restored in turn.
        for state_row in range(3, STATE_LEN):
            tail = tl.load(cache_row + state_row * STATE_STRIDE + offsets, mask=active_channels, other=0).to(
                packed.dtype.element_ty
            )
            tl.store(packed_row + state_row * DIM + channels, tail, mask=in_range)
            tl.store(cache_row + state_row * STATE_STRIDE + offsets, tail, mask=active_channels)
        if tile % channel_tiles == 0:
            # FLA update has no PAD_SLOT_ID argument. The past-end packed row
            # is explicitly supplied as its null_block_id and never accessed.
            tl.store(packed_indices + request, tl.where(active, request, REQUESTS))


def gather_update_conv_state(
    cache: torch.Tensor,
    packed: torch.Tensor,
    x: torch.Tensor,
    cache_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    packed_indices: torch.Tensor,
    *,
    null_block_id: int = 0,
) -> None:
    """Gather BF16 history and commit a width-four single-token state update.

    This replaces gather + post-convolution scatter for *non-speculative*
    decode with max_query_len=1. The native convolution still consumes and
    mutates ``packed``; do not scatter it again. Pass ``packed.shape[0]`` as
    native null_block_id. Prefill, multiple tokens and accepted-token/MTP state
    updates must use the existing two-copy path.

    ``x`` must have exactly one row per request. In that shape, native FLA
    selects fixed-row 2D decode and consumes ``x[request]``; query boundaries
    only mask empty requests in the state-copy path. Compact varlen input
    (fewer rows than requests) must retain the two-copy path.

    Valid active indices must be unique and refer to request-owned mutable
    state, not shared prefix snapshots. Repeated null/padding indices and
    empty segments are allowed. This is also the native in-place update's
    ownership contract; no device-to-host uniqueness scan is performed.
    Every program snapshots the three rows involved in the shift before
    updating them, then restores independent tail rows through BF16. Only
    selected rows are touched, including for a padded page stride or nonzero
    storage offset.
    """
    if cache.ndim != 3 or x.ndim != 2:
        raise ValueError("Expected cache [slots, state_len, dim] and x [tokens, dim]")
    state_len, dim = cache.shape[1:]
    if not 3 <= state_len <= 32 or dim <= 0 or x.shape[1] != dim or x.stride(1) != 1:
        raise ValueError("Single-token width-four state requires 3..32 history rows and contiguous x channels")
    if cache.dtype not in (torch.float32, torch.bfloat16) or x.dtype != torch.bfloat16:
        raise ValueError("Expected FP32/BF16 persistent state and BF16 convolution input")
    requests = cache_indices.numel()
    if cache_indices.ndim != 1 or cache_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("cache_indices must be a one-dimensional integer tensor")
    if x.shape[0] != requests:
        raise ValueError("Fixed-row decode requires exactly one x row per request")
    if query_start_loc.shape != (requests + 1,) or query_start_loc.dtype not in (torch.int32, torch.int64):
        raise ValueError("query_start_loc must have one integer boundary per request plus one")
    if packed.shape != (requests, state_len, dim) or packed.dtype != torch.bfloat16 or not packed.is_contiguous():
        raise ValueError("packed must be a contiguous BF16 [requests, state_len, dim] tensor")
    if packed_indices.shape != (requests,) or packed_indices.dtype != torch.int32 or not packed_indices.is_contiguous():
        raise ValueError("packed_indices must be a contiguous INT32 request vector")
    if any(t.device != cache.device for t in (x, packed, cache_indices, query_start_loc, packed_indices)):
        raise ValueError("All convolution state arguments must be on the same device")
    sd_layout = cache.stride(2) == 1 and cache.stride(1) >= dim
    ds_layout = cache.stride(1) == 1 and cache.stride(2) >= state_len
    row_span = (state_len - 1) * cache.stride(1) + (dim - 1) * cache.stride(2) + 1
    if not (sd_layout or ds_layout) or cache.stride(0) < row_span:
        raise ValueError("State history/channel axes and cache slots must not overlap")
    if requests == 0:
        return
    grid = (min(requests * triton.cdiv(dim, _CONV_STATE_COPY_BLOCK_SIZE), _CONV_STATE_COPY_MAX_PROGRAMS),)
    _gather_update_conv_state_kernel[grid](
        cache,
        packed,
        x,
        cache_indices,
        query_start_loc,
        packed_indices,
        cache.stride(0),
        cache_indices.stride(0),
        query_start_loc.stride(0),
        x.stride(0),
        cache.shape[0],
        x.shape[0],
        requests,
        state_len,
        dim,
        cache.stride(1),
        cache.stride(2),
        NULL_SLOT=null_block_id,
        BLOCK=_CONV_STATE_COPY_BLOCK_SIZE,
    )


@triton.jit
def _copy_conv_state_kernel(
    cache,
    packed,
    cache_indices,
    starts,
    packed_indices,
    cache_stride,
    index_stride,
    num_slots,
    REQUESTS: tl.constexpr,
    STATE_LEN: tl.constexpr,
    DIM: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
    DIM_STRIDE: tl.constexpr,
    WRITE_BACK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    channel_tiles = tl.cdiv(DIM, BLOCK)
    request_tiles = STATE_LEN * channel_tiles
    # Bound the launch grid, including wide TP shards and large batches.
    for tile in range(tl.program_id(0), REQUESTS * request_tiles, tl.num_programs(0)):
        request = tile // request_tiles
        row_tile = tile % request_tiles
        state_row = row_tile // channel_tiles
        channels = row_tile % channel_tiles * BLOCK + tl.arange(0, BLOCK)
        slot = tl.load(cache_indices + request * index_stride).to(tl.int64)
        active = (slot >= 0) & (slot < num_slots) & (tl.load(starts + request + 1) > tl.load(starts + request))
        in_range = channels < DIM
        safe_slot = tl.where(active, slot, 0)
        # Keep the potentially large page address scalar. Within-page offsets
        # fit int32; broadcasting the page address makes every lane use int64.
        cache_row = cache + safe_slot * cache_stride + state_row * STATE_STRIDE
        cache_offsets = channels * DIM_STRIDE
        packed_offsets = request * STATE_LEN * DIM + state_row * DIM + channels
        if WRITE_BACK:
            values = tl.load(packed + packed_offsets, mask=in_range, other=0)
            tl.store(cache_row + cache_offsets, values, mask=active & in_range)
        else:
            values = tl.load(cache_row + cache_offsets, mask=active & in_range, other=0)
            tl.store(packed + packed_offsets, values, mask=in_range)
            if row_tile == 0:
                # The final packed row is a dedicated non-negative null slot.
                # AscendC UPDATE disables null-block filtering for -1.
                tl.store(packed_indices + request, tl.where(active, request, REQUESTS))


def copy_conv_state(
    cache: torch.Tensor,
    packed: torch.Tensor,
    cache_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    packed_indices: torch.Tensor,
    *,
    write_back: bool,
) -> None:
    """Pack/restore request rows; packed has one extra row for the null slot."""
    requests = cache_indices.shape[0]
    if requests == 0:
        return
    state_len, dim = cache.shape[1:]
    grid = (min(requests * state_len * triton.cdiv(dim, _CONV_STATE_COPY_BLOCK_SIZE), _CONV_STATE_COPY_MAX_PROGRAMS),)
    _copy_conv_state_kernel[grid](
        cache,
        packed,
        cache_indices,
        query_start_loc,
        packed_indices,
        cache.stride(0),
        cache_indices.stride(0),
        cache.shape[0],
        requests,
        state_len,
        dim,
        cache.stride(1),
        cache.stride(2),
        WRITE_BACK=write_back,
        BLOCK=_CONV_STATE_COPY_BLOCK_SIZE,
    )
