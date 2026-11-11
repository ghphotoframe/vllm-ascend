# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Copy and zero-pad both MoE inputs in one launch without changing payload bits."""

import torch
from vllm.triton_utils import tl, triton

_TOKEN_PAD_BLOCK_SIZE = 256
_TOKEN_PAD_MAX_PROGRAMS = 65535


@triton.jit
def _pad_tokens_pair_kernel(
    hidden,
    router,
    hidden_out,
    router_out,
    ROWS: tl.constexpr,
    HIDDEN_WIDTH: tl.constexpr,
    ROUTER_WIDTH: tl.constexpr,
    PADDED_ROWS: tl.constexpr,
    HIDDEN_BLOCKS: tl.constexpr,
    TOTAL_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # The branch is uniform within a program. Hidden and router buffers never
    # share a pointer phi or use an in-kernel pointer bitcast on Ascend.
    for block in range(tl.program_id(0), TOTAL_BLOCKS, tl.num_programs(0)):
        if block < HIDDEN_BLOCKS:
            offsets_h = block * BLOCK + tl.arange(0, BLOCK)
            if ROWS > 0:
                source_h = tl.minimum(offsets_h, ROWS * HIDDEN_WIDTH - 1)
                values_h = tl.load(hidden + source_h, offsets_h < ROWS * HIDDEN_WIDTH, other=0)
            else:
                values_h = tl.full((BLOCK,), 0, tl.int16)
            tl.store(hidden_out + offsets_h, values_h, offsets_h < PADDED_ROWS * HIDDEN_WIDTH)
        else:
            offsets_r = (block - HIDDEN_BLOCKS) * BLOCK + tl.arange(0, BLOCK)
            if ROWS > 0:
                source_r = tl.minimum(offsets_r, ROWS * ROUTER_WIDTH - 1)
                values_r = tl.load(router + source_r, offsets_r < ROWS * ROUTER_WIDTH, other=0)
            else:
                values_r = tl.full((BLOCK,), 0, tl.int32)
            tl.store(router_out + offsets_r, values_r, offsets_r < PADDED_ROWS * ROUTER_WIDTH)


def pad_tokens_pair(
    hidden_states: torch.Tensor, router_logits: torch.Tensor, padded_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad contiguous ND BF16/FP32 inputs; the MoE helper selects eligibility."""
    rows, hidden_width = hidden_states.shape
    router_width = router_logits.shape[1]
    hidden_out = hidden_states.new_empty((padded_len, hidden_width))
    router_out = router_logits.new_empty((padded_len, router_width))
    hidden_blocks = triton.cdiv(padded_len * hidden_width, _TOKEN_PAD_BLOCK_SIZE)
    router_blocks = triton.cdiv(padded_len * router_width, _TOKEN_PAD_BLOCK_SIZE)
    blocks = hidden_blocks + router_blocks
    if blocks:
        _pad_tokens_pair_kernel[(min(blocks, _TOKEN_PAD_MAX_PROGRAMS),)](
            hidden_states.view(torch.int16),
            router_logits.view(torch.int32),
            hidden_out.view(torch.int16),
            router_out.view(torch.int32),
            rows,
            hidden_width,
            router_width,
            padded_len,
            hidden_blocks,
            blocks,
            BLOCK=_TOKEN_PAD_BLOCK_SIZE,
        )
    return hidden_out, router_out
