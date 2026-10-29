# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lazy-loaded rotary adapter for the Ling V3 multimodal model."""

import torch
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.rotary_embedding.bailing_mrope import BailingMRotaryEmbedding


@CustomOp.register_oot(name="BailingMRotaryEmbedding")
class AscendBailingMRotaryEmbedding(BailingMRotaryEmbedding):
    """Bailing's alternating spatial/temporal frequencies with shared NPU RoPE."""

    uses_mla_mrope = True

    def select_cos_sin_mla(
        self,
        positions: torch.Tensor,
        output_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Keep the upstream centered-coordinate convention (cos even, sin odd).
        cos, sin = self.select_cos_sin(positions)
        cos = torch.cat((cos, cos), dim=-1)[:, None, None, :]
        sin = torch.cat((sin, sin), dim=-1)[:, None, None, :]
        if output_cache is None:
            return cos, sin
        cos_cache, sin_cache = output_cache
        num_tokens = positions.shape[-1]
        for cache in output_cache:
            if cache.ndim != 4 or cache.shape[1:] != cos.shape[1:] or cache.shape[0] < num_tokens:
                raise ValueError("Bailing MLA RoPE output cache has an incompatible shape or capacity")
            if cache.dtype != cos.dtype or cache.device != cos.device:
                raise ValueError("Bailing MLA RoPE output cache must match the rotary cache dtype and device")
        cos_cache[:num_tokens].copy_(cos)
        sin_cache[:num_tokens].copy_(sin)
        return cos_cache[:num_tokens], sin_cache[:num_tokens]

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if key is None:
            raise ValueError("Bailing M-RoPE requires key states")
        cos, sin = self.select_cos_sin(positions, self._match_cos_sin_cache_dtype(query))
        num_tokens = positions.shape[-1]
        if num_tokens == 0:
            return query, key

        def rotate(x: torch.Tensor) -> torch.Tensor:
            shape = x.shape
            x = x.view(num_tokens, -1, self.head_size)
            rotated = self.apply_rotary_emb(x[..., : self.rotary_dim], cos, sin)
            return torch.cat((rotated, x[..., self.rotary_dim :]), dim=-1).reshape(shape)

        return rotate(query), rotate(key)
