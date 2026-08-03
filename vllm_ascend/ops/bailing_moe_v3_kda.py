#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fla.ops.kda import FusedRMSNormGated
from vllm.model_executor.models.bailing_moe_v3 import BailingMoeV3KimiDeltaAttention
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.ops.triton.fla.utils import clear_ssm_states
from vllm_ascend.ops.triton.kda.kda import rms_norm_gated
from vllm_ascend.ops.triton.mamba.causal_conv1d import causal_conv1d_update_npu

if os.environ.get("ENABLE_PYPTO") == "1":
    from vllm_ascend.ops.pypto.kda.chunk_kda_impl import chunk_kda_wrapper as chunk_kda
    from vllm_ascend.ops.pypto.kda.fused_recurrent_kda_impl import fused_recurrent_kda
else:
    from vllm_ascend.ops.triton.kda.kda import chunk_kda, fused_recurrent_kda


def _to_int64_tuple(tensor: torch.Tensor) -> tuple[int, ...]:
    tensor = tensor.to(torch.int64)
    if tensor.dim() == 0:
        return (tensor.item(),)
    return tuple(tensor.tolist())


def _require_non_spec_prefill_fallback_meta(attn_metadata, field_name: str):
    fallback_meta = getattr(attn_metadata, "non_spec_prefill_fallback_meta", None)
    if fallback_meta is None:
        raise RuntimeError(
            f"Expected attn_metadata.non_spec_prefill_fallback_meta.{field_name} for patched KDA non-spec prefill path."
        )
    return fallback_meta


def _get_non_spec_causal_conv1d_host_args(
    attn_metadata,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    fallback_meta = _require_non_spec_prefill_fallback_meta(attn_metadata, "causal_conv1d")
    causal_conv1d_meta = fallback_meta.causal_conv1d
    return (
        _to_int64_tuple(causal_conv1d_meta.query_start_loc_cpu),
        _to_int64_tuple(causal_conv1d_meta.cache_indices_cpu),
        _to_int64_tuple(causal_conv1d_meta.has_initial_state_cpu),
    )


def _get_non_spec_chunked_prefill_meta(attn_metadata):
    fallback_meta = _require_non_spec_prefill_fallback_meta(attn_metadata, "chunk")
    return fallback_meta.chunk


def _causal_conv1d_prefill(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_state: torch.Tensor,
    host_args: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
) -> torch.Tensor:
    (
        query_start_loc_opt,
        cache_indices_opt,
        initial_state_mode_opt,
    ) = host_args
    weight = weight.transpose(0, 1)
    if weight.dtype != x.dtype:
        weight = weight.to(dtype=x.dtype)
    if bias is not None and bias.dtype != x.dtype:
        bias = bias.to(dtype=x.dtype)
    conv_state_for_op = conv_state
    if conv_state_for_op.dtype != x.dtype:
        conv_state_for_op = conv_state_for_op.to(dtype=x.dtype)

    out = torch.empty_like(x)
    torch.ops._C_ascend.npu_causal_conv1d_custom(
        out,
        x,
        weight,
        conv_state=conv_state_for_op,
        bias_opt=bias,
        query_start_loc_opt=query_start_loc_opt,
        cache_indices_opt=cache_indices_opt,
        initial_state_mode_opt=initial_state_mode_opt,
        num_accepted_tokens_opt=[],
        activation_mode=1,  # silu
        pad_slot_id=PAD_SLOT_ID,
        run_mode=0,
    )
    if conv_state_for_op is not conv_state:
        conv_state.copy_(conv_state_for_op.to(dtype=conv_state.dtype))
    return out


class AscendBailingMoeV3FusedRMSNormGated(FusedRMSNormGated):
    def _forward_decomposed(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
        residual: torch.Tensor | None = None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x_float = x.float()

        residual_out = None
        if residual is not None:
            x_float = x_float + residual.float()
            residual_dtype = torch.float32 if residual_in_fp32 else orig_dtype
            residual_out = x_float.to(residual_dtype)
        elif residual_in_fp32:
            residual_out = x_float

        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        out = x_float * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            out = out * self.weight.float()

        g_float = g.float()
        if self.activation in ("swish", "silu"):
            out = out * F.silu(g_float)
        else:
            out = out * torch.sigmoid(g_float)

        out = out.to(orig_dtype)
        if prenorm:
            if residual_out is None:
                residual_out = x
            return out, residual_out
        return out

    def forward_oot(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
        residual: torch.Tensor | None = None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None and (
            residual_in_fp32 or residual.dtype != x.dtype
        ):
            return self._forward_decomposed(
                x, g, residual, prenorm, residual_in_fp32
            )
        return rms_norm_gated(
            x,
            g,
            self.weight,
            self.bias,
            activation=self.activation,
            residual=residual,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            eps=self.eps,
        )


class AscendBailingMoeV3KimiDeltaAttention(BailingMoeV3KimiDeltaAttention):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        o_norm = self.o_norm
        o_norm_weight = o_norm.weight
        bailing_o_norm = AscendBailingMoeV3FusedRMSNormGated(
            hidden_size=o_norm.hidden_size,
            elementwise_affine=o_norm.elementwise_affine,
            eps=o_norm.eps,
            activation=o_norm.activation,
            device=o_norm_weight.device if o_norm_weight is not None else None,
            dtype=o_norm_weight.dtype if o_norm_weight is not None else None,
        )
        bailing_o_norm.weight = o_norm_weight
        bailing_o_norm.bias = o_norm.bias
        self.o_norm = bailing_o_norm

        self._q_conv1d_update_weight: torch.Tensor | None = None
        self._k_conv1d_update_weight: torch.Tensor | None = None
        self._v_conv1d_update_weight: torch.Tensor | None = None

    @staticmethod
    def _conv1d_weight_2d(conv1d: nn.Module) -> torch.Tensor:
        return conv1d.weight.view(conv1d.weight.size(0), conv1d.weight.size(2))

    def _get_conv1d_update_weight(
        self, cache_name: str, conv1d: nn.Module
    ) -> torch.Tensor:
        weight = self._conv1d_weight_2d(conv1d)
        cached_weight = getattr(self, cache_name)
        if (
            cached_weight is None
            or cached_weight.device != weight.device
            or cached_weight.dtype != weight.dtype
            or cached_weight.shape != (weight.size(1), weight.size(0))
        ):
            cached_weight = weight.transpose(0, 1).contiguous()
            setattr(self, cache_name, cached_weight)
        return cached_weight

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            hidden_states, True
        )
        super().forward(hidden_states, positions, output)

    def _forward(
        self,
        q_proj_states,
        k_proj_states,
        v_proj_states,
        g1,
        beta,
        core_attn_out,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata
        if attn_metadata is None:
            return

        assert isinstance(attn_metadata, dict)
        attn_metadata = attn_metadata[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)
        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices = attn_metadata.spec_state_indices_tensor
        non_spec_state_indices = attn_metadata.non_spec_state_indices_tensor
        num_accepted_tokens = attn_metadata.num_accepted_tokens
        num_actual_tokens = attn_metadata.num_actual_tokens
        conv_state_q, conv_state_k, conv_state_v, recurrent_state = self.kv_cache
        recurrent_state_active = recurrent_state[..., : self.head_dim]
        conv_state_q_transposed = conv_state_q.transpose(-1, -2)
        conv_state_k_transposed = conv_state_k.transpose(-1, -2)
        conv_state_v_transposed = conv_state_v.transpose(-1, -2)

        q_proj_states = q_proj_states[:num_actual_tokens]
        k_proj_states = k_proj_states[:num_actual_tokens]
        v_proj_states = v_proj_states[:num_actual_tokens]
        g1 = g1[:, :num_actual_tokens]
        beta = beta[:, :num_actual_tokens]

        q_conv_weights = self.q_conv1d.weight.view(
            self.q_conv1d.weight.size(0), self.q_conv1d.weight.size(2)
        )
        k_conv_weights = self.k_conv1d.weight.view(
            self.k_conv1d.weight.size(0), self.k_conv1d.weight.size(2)
        )
        v_conv_weights = self.v_conv1d.weight.view(
            self.v_conv1d.weight.size(0), self.v_conv1d.weight.size(2)
        )
        q_conv_update_weight = self._get_conv1d_update_weight(
            "_q_conv1d_update_weight", self.q_conv1d
        )
        k_conv_update_weight = self._get_conv1d_update_weight(
            "_k_conv1d_update_weight", self.k_conv1d
        )
        v_conv_update_weight = self._get_conv1d_update_weight(
            "_v_conv1d_update_weight", self.v_conv1d
        )

        if spec_sequence_masks is not None:
            assert spec_query_start_loc is not None
            assert spec_state_indices is not None
            assert num_accepted_tokens is not None
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                q_proj_states_spec = q_proj_states
                k_proj_states_spec = k_proj_states
                v_proj_states_spec = v_proj_states
                g1_spec = g1
                beta_spec = beta
                q_proj_states_non_spec = None
                k_proj_states_non_spec = None
                v_proj_states_non_spec = None
                g1_non_spec = None
                beta_non_spec = None
            else:
                assert spec_token_indx is not None
                assert non_spec_token_indx is not None
                q_proj_states_spec = q_proj_states.index_select(0, spec_token_indx)
                k_proj_states_spec = k_proj_states.index_select(0, spec_token_indx)
                v_proj_states_spec = v_proj_states.index_select(0, spec_token_indx)
                g1_spec = g1.index_select(1, spec_token_indx)
                beta_spec = beta.index_select(1, spec_token_indx)
                q_proj_states_non_spec = q_proj_states.index_select(
                    0, non_spec_token_indx
                )
                k_proj_states_non_spec = k_proj_states.index_select(
                    0, non_spec_token_indx
                )
                v_proj_states_non_spec = v_proj_states.index_select(
                    0, non_spec_token_indx
                )
                g1_non_spec = g1.index_select(1, non_spec_token_indx)
                beta_non_spec = beta.index_select(1, non_spec_token_indx)
        else:
            q_proj_states_spec = None
            k_proj_states_spec = None
            v_proj_states_spec = None
            g1_spec = None
            beta_spec = None
            q_proj_states_non_spec = q_proj_states
            k_proj_states_non_spec = k_proj_states
            v_proj_states_non_spec = v_proj_states
            g1_non_spec = g1
            beta_non_spec = beta

        def _causal_conv1d_spec(
            x: torch.Tensor,
            conv_state: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor | None,
        ) -> torch.Tensor:
            assert spec_query_start_loc is not None
            assert spec_state_indices is not None
            assert num_accepted_tokens is not None
            return causal_conv1d_update_npu(
                x,
                conv_state,
                weight,
                bias,
                activation="silu",
                conv_state_indices=spec_state_indices[:, 0][
                    : attn_metadata.num_spec_decodes
                ],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices.size(-1),
                weight_is_transposed=True,
                validate_data=False,
            )

        if spec_sequence_masks is not None:
            assert q_proj_states_spec is not None
            assert k_proj_states_spec is not None
            assert v_proj_states_spec is not None
            q_spec = _causal_conv1d_spec(
                q_proj_states_spec,
                conv_state_q_transposed,
                q_conv_update_weight,
                self.q_conv1d.bias,
            )
            k_spec = _causal_conv1d_spec(
                k_proj_states_spec,
                conv_state_k_transposed,
                k_conv_update_weight,
                self.k_conv1d.bias,
            )
            v_spec = _causal_conv1d_spec(
                v_proj_states_spec,
                conv_state_v_transposed,
                v_conv_update_weight,
                self.v_conv1d.bias,
            )
        else:
            q_spec = None
            k_spec = None
            v_spec = None

        if attn_metadata.num_prefills > 0:
            assert q_proj_states_non_spec is not None
            assert k_proj_states_non_spec is not None
            assert v_proj_states_non_spec is not None
            causal_conv1d_host_args = _get_non_spec_causal_conv1d_host_args(
                attn_metadata
            )
            q = _causal_conv1d_prefill(
                q_proj_states_non_spec,
                q_conv_weights,
                self.q_conv1d.bias,
                conv_state_q,
                causal_conv1d_host_args,
            )
            k = _causal_conv1d_prefill(
                k_proj_states_non_spec,
                k_conv_weights,
                self.k_conv1d.bias,
                conv_state_k,
                causal_conv1d_host_args,
            )
            v = _causal_conv1d_prefill(
                v_proj_states_non_spec,
                v_conv_weights,
                self.v_conv1d.bias,
                conv_state_v,
                causal_conv1d_host_args,
            )
        elif attn_metadata.num_decodes > 0:
            assert q_proj_states_non_spec is not None
            assert k_proj_states_non_spec is not None
            assert v_proj_states_non_spec is not None
            assert non_spec_state_indices is not None
            decode_indices = non_spec_state_indices[:num_actual_tokens]
            q = causal_conv1d_update_npu(
                q_proj_states_non_spec,
                conv_state_q_transposed,
                q_conv_update_weight,
                self.q_conv1d.bias,
                activation="silu",
                conv_state_indices=decode_indices,
                weight_is_transposed=True,
                validate_data=True,
            )
            k = causal_conv1d_update_npu(
                k_proj_states_non_spec,
                conv_state_k_transposed,
                k_conv_update_weight,
                self.k_conv1d.bias,
                activation="silu",
                conv_state_indices=decode_indices,
                weight_is_transposed=True,
                validate_data=True,
            )
            v = causal_conv1d_update_npu(
                v_proj_states_non_spec,
                conv_state_v_transposed,
                v_conv_update_weight,
                self.v_conv1d.bias,
                activation="silu",
                conv_state_indices=decode_indices,
                weight_is_transposed=True,
                validate_data=True,
            )
        else:
            q = None
            k = None
            v = None

        if q_spec is not None:
            q_spec, k_spec, v_spec = map(
                lambda x: rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim),
                (q_spec, k_spec, v_spec),
            )
        if q is not None:
            q, k, v = map(
                lambda x: rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim),
                (q, k, v),
            )

        if spec_sequence_masks is not None:
            assert q_spec is not None
            assert k_spec is not None
            assert v_spec is not None
            assert g1_spec is not None
            assert beta_spec is not None
            assert spec_query_start_loc is not None
            assert spec_state_indices is not None
            assert num_accepted_tokens is not None
            out_spec, _ = fused_recurrent_kda(
                q=q_spec,
                k=k_spec,
                v=v_spec,
                g=g1_spec,
                beta=beta_spec,
                initial_state=recurrent_state_active,
                use_qk_l2norm_in_kernel=True,
                safe_gate=self.safe_gate,
                lower_bound=self.lower_bound,
                cu_seqlens=spec_query_start_loc[
                    : attn_metadata.num_spec_decodes + 1
                ],
                ssm_state_indices=spec_state_indices,
                num_accepted_tokens=num_accepted_tokens,
            )
        else:
            out_spec = None

        if attn_metadata.num_prefills > 0:
            assert q is not None
            assert k is not None
            assert v is not None
            assert g1_non_spec is not None
            assert beta_non_spec is not None
            assert non_spec_query_start_loc is not None
            assert non_spec_state_indices is not None
            initial_state = recurrent_state_active[non_spec_state_indices].contiguous()
            clear_ssm_states(initial_state, has_initial_state)
            out, last_state = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g1_non_spec,
                beta=beta_non_spec,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                safe_gate=self.safe_gate,
                lower_bound=self.lower_bound,
                cu_seqlens=non_spec_query_start_loc,
                prebuilt_meta=_get_non_spec_chunked_prefill_meta(attn_metadata),
            )
            recurrent_state_active[non_spec_state_indices] = last_state
        elif attn_metadata.num_decodes > 0:
            assert q is not None
            assert k is not None
            assert v is not None
            assert g1_non_spec is not None
            assert beta_non_spec is not None
            assert non_spec_query_start_loc is not None
            out, _ = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g1_non_spec,
                beta=beta_non_spec,
                initial_state=recurrent_state_active,
                use_qk_l2norm_in_kernel=True,
                safe_gate=self.safe_gate,
                lower_bound=self.lower_bound,
                cu_seqlens=non_spec_query_start_loc[
                    : attn_metadata.num_decodes + 1
                ],
                ssm_state_indices=non_spec_state_indices,
            )
        else:
            out = None

        if spec_sequence_masks is not None and out is not None:
            assert out_spec is not None
            assert spec_token_indx is not None
            assert non_spec_token_indx is not None
            merged_out = torch.empty(
                (1, num_actual_tokens, *out_spec.shape[2:]),
                dtype=out.dtype,
                device=out.device,
            )
            merged_out.index_copy_(1, spec_token_indx, out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, out)
            core_attn_out[0, :num_actual_tokens] = merged_out[0, :num_actual_tokens]
        elif spec_sequence_masks is not None:
            assert out_spec is not None
            core_attn_out[0, :num_actual_tokens] = out_spec[0, :num_actual_tokens]
        else:
            assert out is not None
            core_attn_out[0, :num_actual_tokens] = out[0, :num_actual_tokens]
