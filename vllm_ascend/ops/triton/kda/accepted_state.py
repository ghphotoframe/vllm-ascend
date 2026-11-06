# SPDX-License-Identifier: Apache-2.0
"""Commit a prior accepted KDA state before a shorter verification query.

AscendC currently bounds the previous accepted count by the *current* query
length. Rebase only this boundary case onto slot zero and pass a local count
of one. Current-step destination slots and host acceptance metadata are kept.
"""

import torch
from vllm.triton_utils import tl, triton

_KDA_MAX_VERIFY_TOKENS = 8
_STATE_COPY_BLOCK_SIZE = 4096


@triton.jit
def _rebase_short_query_kernel(
    state,
    slots,
    query_start,
    accepted,
    adjusted,
    state_stride,
    capacity,
    payload,
    slots_row_stride,
    slots_col_stride,
    slot_columns,
    MAX_STEP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    old_accepted = tl.load(accepted + row).to(tl.int64)
    length = (tl.load(query_start + row + 1) - tl.load(query_start + row)).to(tl.int64)
    source_column = tl.minimum(tl.maximum(old_accepted - 1, 0), slot_columns - 1)
    source = tl.load(slots + row * slots_row_stride + source_column * slots_col_stride).to(tl.int64)
    dest = tl.load(slots + row * slots_row_stride).to(tl.int64)
    active = (
        (length > 0)
        & (length <= MAX_STEP)
        & (old_accepted > length)
        & (old_accepted <= slot_columns)
        & (old_accepted <= MAX_STEP)
        & (source >= 0)
        & (source < capacity)
        & (dest >= 0)
        & (dest < capacity)
    )
    # Match the original kernel's validation of every current destination.
    for step in range(MAX_STEP):
        index = tl.load(slots + row * slots_row_stride + step * slots_col_stride, mask=step < length, other=0).to(
            tl.int64
        )
        active = active & ((step >= length) | ((index >= 0) & (index < capacity)))
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(state + source * state_stride + offsets, mask=active & (offsets < payload), other=0)
    tl.store(state + dest * state_stride + offsets, values, mask=active & (offsets < payload))
    if tile == 0:
        tl.store(adjusted + row, tl.where(active, 1, old_accepted))


def rebase_short_kda_queries(
    state: torch.Tensor,
    state_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
) -> torch.Tensor:
    """Device-only state commit; never truncate or disable speculative tokens.

    Cache slots written by distinct sequences must be disjoint, as required by
    the original in-place recurrent kernel. Invalid/padding rows remain intact.
    """
    if state_indices.ndim != 2 or num_accepted_tokens.numel() == 0:
        return num_accepted_tokens
    rows, columns = state_indices.shape
    if columns == 0 or rows != num_accepted_tokens.numel() or query_start_loc.numel() != rows + 1:
        raise ValueError("KDA acceptance metadata shapes do not match")
    if state.ndim != 4 or state.shape[0] == 0 or state.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("KDA rebase requires a four-dimensional FP32/BF16 cache")
    if any(x.dtype not in (torch.int32, torch.int64) for x in (state_indices, query_start_loc, num_accepted_tokens)):
        raise ValueError("KDA metadata must use INT32 or INT64")
    inner = 1
    for size, stride in zip(reversed(state.shape[1:]), reversed(state.stride()[1:])):
        if size <= 0 or (size > 1 and stride != inner):
            raise ValueError("KDA rebase requires a dense inner state payload")
        inner *= size
    if state.stride(0) < inner:
        raise ValueError("KDA cache pages must not overlap")
    if (
        num_accepted_tokens.ndim != 1
        or num_accepted_tokens.stride(0) != 1
        or query_start_loc.ndim != 1
        or query_start_loc.stride(0) != 1
    ):
        raise ValueError("KDA acceptance and query starts must be contiguous vectors")
    if any(x.device != state.device for x in (state_indices, query_start_loc, num_accepted_tokens)):
        raise ValueError("KDA metadata must be on the cache device")
    adjusted = torch.empty_like(num_accepted_tokens)
    _rebase_short_query_kernel[(rows, triton.cdiv(inner, _STATE_COPY_BLOCK_SIZE))](
        state,
        state_indices,
        query_start_loc,
        num_accepted_tokens,
        adjusted,
        state.stride(0),
        state.shape[0],
        inner,
        state_indices.stride(0),
        state_indices.stride(1),
        columns,
        MAX_STEP=min(columns, _KDA_MAX_VERIFY_TOKENS),
        BLOCK=_STATE_COPY_BLOCK_SIZE,
    )
    return adjusted
