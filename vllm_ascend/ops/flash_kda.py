# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapter for the optional A5 PyPTO FlashKDA V1 distribution.

Place the vendor's ``flash_kda/`` directory on PYTHONPATH. Do not import PyPTO
until an opted-in worker actually executes prefill. This adapter implements
the source-verified V1 ABI; it does not vendor or install the external operator.
"""

from collections.abc import Sequence
from numbers import Integral

import torch

from vllm_ascend.device import HardwareCapability, get_current_hardware_profile

FLASH_KDA_HEAD_DIM = 128
FLASH_KDA_MAX_KEY_HEADS = 64
FLASH_KDA_MAX_VALUE_HEADS = 96
FLASH_KDA_MAX_SEQUENCES = 64


def _validate_inputs(q, k, v, raw_gate, beta, initial_state, cu_seqlens, a_log, dt_bias, lower_bound):
    if not get_current_hardware_profile().supports(HardwareCapability.FLASH_KDA_PREFILL):
        raise RuntimeError("FlashKDA is supported only on A5.")
    if lower_bound is None or not -5.0 <= lower_bound <= 0.0:
        raise ValueError("FlashKDA requires safe_gate=True and lower_bound in [-5, 0].")
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[-1] != FLASH_KDA_HEAD_DIM:
        raise ValueError("FlashKDA requires packed BSND q with B=1 and D=128.")
    if k.shape != q.shape or v.ndim != 4 or v.shape[:2] != q.shape[:2] or v.shape[-1] != FLASH_KDA_HEAD_DIM:
        raise ValueError("FlashKDA requires matching q/k and packed v with D=128.")
    num_tokens, num_key_heads, num_value_heads = q.shape[1], q.shape[2], v.shape[2]
    if num_tokens <= 0 or num_key_heads <= 0 or num_value_heads <= 0 or num_value_heads % num_key_heads:
        raise ValueError("FlashKDA requires positive tokens/heads and Nv % Nk == 0.")
    if num_key_heads > FLASH_KDA_MAX_KEY_HEADS or num_value_heads > FLASH_KDA_MAX_VALUE_HEADS:
        raise ValueError("FlashKDA V1 supports at most Nk=64 and Nv=96.")
    if raw_gate.shape != v.shape or beta.shape != v.shape[:-1]:
        raise ValueError("FlashKDA requires gate [1,T,Nv,128] and raw beta logits [1,T,Nv].")
    if any(tensor.dtype != torch.bfloat16 for tensor in (q, k, v, raw_gate, beta)):
        raise ValueError("FlashKDA requires BF16 q/k/v/g and raw beta logits (not sigmoid(beta)).")
    if q.device.type != "npu" or any(
        tensor.device != q.device for tensor in (k, v, raw_gate, beta, initial_state, a_log, dt_bias)
    ):
        raise ValueError("FlashKDA requires all input tensors on the same NPU.")
    # The GDN builder already filters empty sequences on the host. Validate
    # those descriptors without tensor.item()/tolist() device synchronization.
    if isinstance(cu_seqlens, torch.Tensor):
        raise ValueError("FlashKDA adapter requires GDN host cu_seqlens, not a device tensor.")
    if len(cu_seqlens) - 1 > FLASH_KDA_MAX_SEQUENCES:
        raise ValueError("FlashKDA V1 supports at most 64 sequences per prefill call.")
    if (
        len(cu_seqlens) < 2
        or any(not isinstance(offset, Integral) for offset in cu_seqlens)
        or cu_seqlens[0] != 0
        or cu_seqlens[-1] != num_tokens
        or any(end <= start for start, end in zip(cu_seqlens, cu_seqlens[1:]))
    ):
        raise ValueError("FlashKDA cu_seqlens must span all packed tokens with no empty sequences.")
    if initial_state.shape != (len(cu_seqlens) - 1, num_value_heads, FLASH_KDA_HEAD_DIM, FLASH_KDA_HEAD_DIM):
        raise ValueError("FlashKDA initial_state must have shape [num_sequences,Nv,Dv,Dk].")
    if initial_state.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("FlashKDA initial_state must use FP32 or BF16 cache storage.")
    if a_log.numel() != num_value_heads or dt_bias.numel() != num_value_heads * FLASH_KDA_HEAD_DIM:
        raise ValueError("FlashKDA requires A_log [Nv] and dt_bias [Nv,128].")


def run_flash_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: Sequence[int],
    chunk_indices: Sequence[int],
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    lower_bound: float | None,
    cu_seqlens_device: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Consume raw BF16 beta logits; return BF16 output and FP32 VK state.

    FlashKDA fuses Q/K L2 norm, beta sigmoid and bounded gate conversion.
    Always use varlen, including an unaligned single sequence; the vendor
    kernel handles tail padding. Cache gather/scatter remain caller-owned.
    The production caller passes the batch's compacted INT64 device cu tensor;
    its values must match the validated host descriptors. Only tensor metadata
    is inspected here: reading device values would synchronize every layer.
    Direct standalone callers may omit it to construct their own descriptors.
    """
    _validate_inputs(q, k, v, raw_gate, beta, initial_state, cu_seqlens, a_log, dt_bias, lower_bound)
    if cu_seqlens_device is not None and (
        not isinstance(cu_seqlens_device, torch.Tensor)
        or cu_seqlens_device.device != q.device
        or cu_seqlens_device.dtype != torch.int64
        or cu_seqlens_device.ndim != 1
        or cu_seqlens_device.numel() != len(cu_seqlens)
        or not cu_seqlens_device.is_contiguous()
    ):
        raise ValueError("FlashKDA prepared cu_seqlens must be contiguous INT64 [N+1] on the input NPU.")
    # Lazy loading isolates this optional dependency from default/A2/A3 use.
    try:
        from flash_kda.flash_kda import flash_kda_wrapper
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "FlashKDA was enabled but its optional dependency is missing. Install the compatible "
            "PyPTO distribution and put the vendor flash_kda/ directory on PYTHONPATH "
            "(from flash_kda.flash_kda import flash_kda_wrapper), or set VLLM_ASCEND_ENABLE_FLASH_KDA=0."
        ) from exc

    del chunk_indices  # FlashKDA builds its own chunk descriptors.
    num_value_heads = v.shape[2]
    if cu_seqlens_device is None:
        # Compatibility for direct operator tests/tools. Serving passes the
        # batch-owned tensor and performs no per-layer allocation/H2D here.
        cu_seqlens_device = torch.tensor(cu_seqlens, dtype=torch.int64, device=q.device)
    return flash_kda_wrapper(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        raw_gate.contiguous(),
        beta.contiguous(),
        scale=FLASH_KDA_HEAD_DIM**-0.5,
        initial_state=initial_state.float().contiguous(),
        a_log=a_log.reshape(num_value_heads).float().contiguous(),
        dt_bias=dt_bias.reshape(num_value_heads, FLASH_KDA_HEAD_DIM).float().contiguous(),
        lower_bound=lower_bound,
        layout_qkv="BSND",
        # V1's TND kernels consume INT64; avoid its per-call INT32->INT64 cast.
        cu_seqlens=cu_seqlens_device,
    )
