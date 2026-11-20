# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep the already-host-side cumulative endpoints for Bailing eager vision."""

from contextvars import ContextVar
from functools import wraps

import numpy as np
import torch
from vllm.config import get_current_vllm_config_or_none
from vllm.logger import logger
from vllm.model_executor.models.bailing_moe_v3_vl import BailingMoeV3VisionTransformer
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from vllm_ascend.ops.mm_encoder_attention import AscendMMEncoderAttention
from vllm_ascend.worker.encoder_acl_graph import get_encoder_forward_context

_GRAPH_METADATA_ARGUMENTS = ("max_batch_size", "max_frames_per_batch", "max_seqlen_override")

_bailing_vision_metadata_scope: ContextVar[bool] = ContextVar(
    "bailing_vision_metadata_scope",
    default=False,
)
_original_init = BailingMoeV3VisionTransformer.__init__
_original_prepare_encoder_metadata = BailingMoeV3VisionTransformer.prepare_encoder_metadata
_original_recompute_cu_seqlens = AscendMMEncoderAttention.maybe_recompute_cu_seqlens


def _all_layers_use_ascend_fia(blocks) -> bool:
    for block in blocks:
        attention = getattr(getattr(block, "attn", None), "attn", None)
        if (
            type(attention) is not AscendMMEncoderAttention
            or getattr(attention, "_forward_method", None) != attention.forward_oot
        ):
            return False
    return len(blocks) > 0


@wraps(_original_init)
def _init_with_host_metadata_preference(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    config = get_current_vllm_config_or_none()
    self._use_host_cu_seqlens = (
        config is not None
        and not config.compilation_config.compile_mm_encoder
        and not config.compilation_config.cudagraph_mm_encoder
        and _all_layers_use_ascend_fia(self.blocks)
    )


@wraps(_original_prepare_encoder_metadata)
def _prepare_encoder_metadata(self, grid_thw_list, **kwargs):
    if (
        type(self) is not BailingMoeV3VisionTransformer
        or not self._use_host_cu_seqlens
        or self.dtype != torch.bfloat16
        or self.attn_backend != AttentionBackendEnum.TORCH_SDPA
        or torch.compiler.is_compiling()
        or get_encoder_forward_context().capturing
        or any(kwargs.get(key) is not None for key in _GRAPH_METADATA_ARGUMENTS)
    ):
        return _original_prepare_encoder_metadata(self, grid_thw_list, **kwargs)

    scope = _bailing_vision_metadata_scope.set(True)
    try:
        return _original_prepare_encoder_metadata(self, grid_thw_list, **kwargs)
    finally:
        _bailing_vision_metadata_scope.reset(scope)


def _maybe_recompute_cu_seqlens(
    cls,
    attn_backend: AttentionBackendEnum,
    cu_seqlens: np.ndarray,
    hidden_size: int,
    tp_size: int,
    device: torch.device,
    fp8_padded_hidden_size: int | None = None,
) -> torch.Tensor:
    if (
        cls is AscendMMEncoderAttention
        and _bailing_vision_metadata_scope.get()
        and attn_backend == AttentionBackendEnum.TORCH_SDPA
        and device.type == "npu"
        and cu_seqlens.ndim == 1
        and cu_seqlens.dtype == np.int32
        and fp8_padded_hidden_size is None
    ):
        logger.info_once("Bailing eager ViT uses host cu_seqlens for FIA")
        return torch.from_numpy(cu_seqlens)

    return _original_recompute_cu_seqlens(
        attn_backend,
        cu_seqlens,
        hidden_size,
        tp_size,
        device,
        fp8_padded_hidden_size=fp8_padded_hidden_size,
    )


BailingMoeV3VisionTransformer.__init__ = _init_with_host_metadata_preference
BailingMoeV3VisionTransformer.prepare_encoder_metadata = _prepare_encoder_metadata
AscendMMEncoderAttention.maybe_recompute_cu_seqlens = classmethod(_maybe_recompute_cu_seqlens)
