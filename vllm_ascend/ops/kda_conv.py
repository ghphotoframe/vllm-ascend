# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import torch
from fla_npu.ops.ascendc import causal_conv1d_fn, causal_conv1d_update
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID


def pack_kda_conv_weights(
    weights: Sequence[torch.Tensor],
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Pack combined QKV or separate Q/K/V weights for FLA causal convolution.

    Canonical weights have shape ``[channels, 1, width]``. The result is
    contiguous ``[width, 3 * local_projection_size]`` in Q/K/V order.
    Callers own the packed storage and its post-load/reload lifecycle.
    """
    if len(weights) not in (1, 3):
        raise ValueError("Expected one combined QKV weight or three Q/K/V weights")
    if any(weight.ndim != 3 or weight.shape[1] != 1 for weight in weights):
        raise ValueError("KDA convolution weights must have shape [channels, 1, width]")
    if any(weight.shape != weights[0].shape for weight in weights[1:]):
        raise ValueError("Separate KDA Q/K/V convolution weights must have matching shapes")

    combined = weights[0] if len(weights) == 1 else torch.cat(tuple(weights), dim=0)
    return combined.squeeze(1).transpose(0, 1).to(device=device, dtype=dtype).contiguous()


def native_conv_state_supported(state: torch.Tensor, input_dtype: torch.dtype) -> bool:
    """FLA accepts same-dtype SD states with non-overlapping outer strides."""
    if state.ndim != 3 or input_dtype not in (torch.bfloat16, torch.float16) or state.dtype != input_dtype:
        return False
    slots, history, channels = state.shape
    stride_slot, stride_history, stride_channel = state.stride()
    return (
        min(slots, history, channels) > 0
        and stride_channel == 1
        and stride_history >= channels
        and stride_slot >= (history - 1) * stride_history + channels
    )


def run_causal_conv1d(
    mixed_qkv: torch.Tensor,
    conv_weights_t: torch.Tensor,
    conv_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    initial_state_mode: torch.Tensor | None,
    *,
    run_mode: int,
    max_query_len: int = -1,
    num_accepted_tokens: torch.Tensor | None = None,
    null_block_id: int = NULL_BLOCK_ID,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Shared Kimi/Ling native call; callers own state and index preparation.

    Decode callers needing deterministic padding supply a zeroed output.
    Otherwise retain Kimi's existing allocation and native update behavior.
    """
    if run_mode == 0:
        return causal_conv1d_fn(
            mixed_qkv,
            conv_weights_t,
            None,
            conv_states=conv_state,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=initial_state_mode,
            activation="silu",
            pad_slot_id=PAD_SLOT_ID,
            null_block_id=null_block_id,
        )
    if run_mode != 1:
        raise ValueError(f"Unsupported causal_conv1d run_mode: {run_mode}")
    return causal_conv1d_update(
        mixed_qkv,
        conv_state,
        conv_weights_t,
        bias=None,
        activation="silu",
        conv_state_indices=cache_indices,
        num_accepted_tokens=num_accepted_tokens,
        query_start_loc=query_start_loc,
        max_query_len=max_query_len,
        null_block_id=null_block_id,
        out=torch.empty_like(mixed_qkv) if out is None else out,
    )
