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

import torch
from einops import rearrange
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.bailing_moe_v3 import BailingMoeV3KimiDeltaAttention
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from vllm_ascend.ops.triton.fla.kda import chunk_kda, fused_recurrent_kda


class AscendBailingMoeV3KimiDeltaAttention(BailingMoeV3KimiDeltaAttention):

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
        query_start_loc = attn_metadata.non_spec_query_start_loc
        state_indices = attn_metadata.non_spec_state_indices_tensor
        num_actual_tokens = attn_metadata.num_actual_tokens
        conv_state_q, conv_state_k, conv_state_v, recurrent_state = (
            self.kv_cache
        )
        recurrent_state_active = recurrent_state[..., : self.head_dim]
        conv_state_q = conv_state_q.transpose(-1, -2)
        conv_state_k = conv_state_k.transpose(-1, -2)
        conv_state_v = conv_state_v.transpose(-1, -2)

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
        if attn_metadata.num_prefills > 0:
            q = causal_conv1d_fn(
                q_proj_states.transpose(0, 1),
                q_conv_weights,
                self.q_conv1d.bias,
                activation="silu",
                conv_states=conv_state_q,
                has_initial_state=has_initial_state,
                cache_indices=state_indices,
                query_start_loc=query_start_loc,
                metadata=attn_metadata,
            ).transpose(0, 1)
            k = causal_conv1d_fn(
                k_proj_states.transpose(0, 1),
                k_conv_weights,
                self.k_conv1d.bias,
                activation="silu",
                conv_states=conv_state_k,
                has_initial_state=has_initial_state,
                cache_indices=state_indices,
                query_start_loc=query_start_loc,
                metadata=attn_metadata,
            ).transpose(0, 1)
            v = causal_conv1d_fn(
                v_proj_states.transpose(0, 1),
                v_conv_weights,
                self.v_conv1d.bias,
                activation="silu",
                conv_states=conv_state_v,
                has_initial_state=has_initial_state,
                cache_indices=state_indices,
                query_start_loc=query_start_loc,
                metadata=attn_metadata,
            ).transpose(0, 1)
        else:
            decode_indices = state_indices[:num_actual_tokens]
            q = causal_conv1d_update(
                q_proj_states,
                conv_state_q,
                q_conv_weights,
                self.q_conv1d.bias,
                activation="silu",
                conv_state_indices=decode_indices,
                validate_data=True,
            )
            k = causal_conv1d_update(
                k_proj_states,
                conv_state_k,
                k_conv_weights,
                self.k_conv1d.bias,
                activation="silu",
                conv_state_indices=decode_indices,
                validate_data=True,
            )
            v = causal_conv1d_update(
                v_proj_states,
                conv_state_v,
                v_conv_weights,
                self.v_conv1d.bias,
                activation="silu",
                conv_state_indices=decode_indices,
                validate_data=True,
            )

        q, k, v = map(
            lambda x: rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim), (q, k, v)
        )
        if attn_metadata.num_prefills > 0:
            zero_idx = state_indices[~has_initial_state]
            recurrent_state[zero_idx] = 0
            initial_state = recurrent_state_active[state_indices].contiguous()
            out, last_state = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g1,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                safe_gate=self.safe_gate,
                lower_bound=self.lower_bound,
                cu_seqlens=query_start_loc,
            )
            recurrent_state_active[state_indices] = last_state
        else:
            out, _ = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g1,
                beta=beta,
                initial_state=recurrent_state_active,
                use_qk_l2norm_in_kernel=True,
                safe_gate=self.safe_gate,
                lower_bound=self.lower_bound,
                cu_seqlens=query_start_loc[: attn_metadata.num_decodes + 1],
                ssm_state_indices=state_indices,
            )
        core_attn_out[0, :num_actual_tokens] = out[0, :num_actual_tokens]