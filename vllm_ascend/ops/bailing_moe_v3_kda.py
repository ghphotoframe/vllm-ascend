# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend backend for the Bailing/Ling V3 Kimi Delta Attention layer."""

import torch
import torch_npu
from einops import rearrange
from torch._subclasses.fake_tensor import FakeTensor
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import get_current_vllm_config_or_none
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.models.bailing_moe_v3 import BailingMoeV3KimiDeltaAttention
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID

from vllm_ascend.ops.gdn_attn_builder import AscendFlashKDAAttentionBackend, AscendGDNAttentionBackend
from vllm_ascend.ops.kda import run_recurrent_kda
from vllm_ascend.ops.kda_backend import FLASH_KDA_ENABLED, run_chunk_kda
from vllm_ascend.ops.kda_conv import (
    native_conv_state_supported,
    pack_kda_conv_weights,
    prepare_native_conv_decode,
    run_causal_conv1d,
)
from vllm_ascend.ops.triton.kda.accepted_state import rebase_short_kda_queries
from vllm_ascend.ops.triton.kda.conv_state import copy_conv_state, gather_update_conv_state
from vllm_ascend.utils import maybe_trans_nz

_KDA_MAX_RECURRENT_TOKENS = 8
_PACKED_CONV_WEIGHT_NAME = "ascend_conv1d_weight"
_PACKED_INPUT_PROJECTION_NAME = "ascend_input_projection_weight"
_KDA_INPUT_PROJECTION_ALIGNMENT = 16
_KDA_ND_WEIGHT_FORMAT = 2


def _input_projection_weight_format(weight: torch.Tensor) -> int | None:
    return int(torch_npu.get_npu_format(weight)) if weight.device.type == "npu" else None


def _input_projection_weight_nd(weight: torch.Tensor) -> torch.Tensor:
    return torch_npu.npu_format_cast(weight, _KDA_ND_WEIGHT_FORMAT) if weight.device.type == "npu" else weight


def _zero_padded_spec_output(output: torch.Tensor, query_start_loc: torch.Tensor) -> torch.Tensor:
    """Clear graph-padding rows that the recurrent kernel does not write."""
    token_indices = torch.arange(output.shape[1], dtype=query_start_loc.dtype, device=output.device)
    valid_tokens = token_indices < query_start_loc[-1]
    return torch.where(valid_tokens.view(1, -1, 1, 1), output, 0.0)


def _contiguous_packed_qkv(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, allow_row_gaps: bool = False
) -> torch.Tensor:
    """Reuse adjacent projection slices when permitted by the caller.

    A projection may have columns after Q/K/V, creating row gaps. Only the
    native fixed-decode prepare path may defer compaction of such a view;
    native convolution itself still receives a dense buffer. A single row
    is already contiguous. All checks use tensor metadata, without caching.
    """
    if (
        q.ndim == k.ndim == v.ndim == 2
        and q.shape == k.shape == v.shape
        and q.shape[0] > 0
        and q.dtype == k.dtype == v.dtype
        and q.device == k.device == v.device
        and not q.is_meta
        and not any(isinstance(tensor, FakeTensor) for tensor in (q, k, v))
        and q.stride() == k.stride() == v.stride()
        and q.stride(-1) == 1
    ):
        width = q.shape[1]
        if (
            k.storage_offset() == q.storage_offset() + width
            and v.storage_offset() == q.storage_offset() + 2 * width
            and q.untyped_storage().data_ptr() == k.untyped_storage().data_ptr() == v.untyped_storage().data_ptr()
        ):
            packed = q.as_strided((q.shape[0], 3 * width), q.stride(), storage_offset=q.storage_offset())
            if packed.is_contiguous() or (allow_row_gaps and q.stride(0) >= 3 * width):
                return packed
    return torch.cat((q, k, v), dim=-1)


def _native_decode_prepare_compatible(
    num_tokens: int,
    input_dtype: torch.dtype,
    conv_state: torch.Tensor,
    cache_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_speculative_tokens: int,
) -> bool:
    return (
        native_conv_state_supported(conv_state, input_dtype)
        and num_speculative_tokens == 0
        and input_dtype == torch.bfloat16
        and cache_indices.ndim in (1, 2)
        and cache_indices.shape[0] == num_tokens
        and (cache_indices.ndim == 1 or cache_indices.shape[1] > 0)
        and cache_indices.dtype == torch.int32
        and query_start_loc.dtype == torch.int32
        and query_start_loc.shape == (num_tokens + 1,)
    )


@PluggableLayer.register_oot(name="BailingMoeV3KimiDeltaAttention")
class AscendBailingMoeV3KimiDeltaAttention(BailingMoeV3KimiDeltaAttention):
    """Run Bailing/Ling V3 KDA with the current Ascend GDN metadata and ops."""

    # Worker startup prepares this layer after KV-cache binding on every device.
    _requires_kda_state_copy = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._ascend_kda_state_copy = None
        # The device kernel skips sequences longer than MAX_MTP without a
        # host-side error, so reject an unsupported verification width here.
        if self.num_speculative_tokens + 1 > _KDA_MAX_RECURRENT_TOKENS:
            raise ValueError(
                "Bailing/Ling V3 Ascend KDA supports at most "
                f"{_KDA_MAX_RECURRENT_TOKENS - 1} speculative tokens, got {self.num_speculative_tokens}."
            )
        self._conv_state_dim_first = is_conv_state_dim_first()
        # Derived kernel storage is not a checkpoint parameter. A persistent
        # parameter here would fail the unquantized loader's strict tracking.
        self.register_buffer(
            _PACKED_CONV_WEIGHT_NAME,
            torch.empty(
                self.conv_size,
                3 * self.projection_size_per_partition,
                dtype=self.model_config.dtype if self.model_config is not None else torch.get_default_dtype(),
                device=self.q_conv1d.weight.device,
            ),
            persistent=False,
        )
        self._ascend_conv_weights_ready = False
        self._ascend_conv_weights_initialized = False
        config = kwargs.get("config", args[0] if args else None)
        fuse_inputs = getattr(config, "fuse_kda_input_projections", False)
        if type(fuse_inputs) is not bool:
            raise ValueError("fuse_kda_input_projections must be a boolean")
        self._ascend_fuse_input_projections = fuse_inputs
        if fuse_inputs:
            self._validate_input_projection_fusion()
            # Keep canonical qkvb/f/g checkpoint parameters. The opt-in copy
            # costs about 440 MiB/rank for the TP8 Ling-3.0 35-layer checkpoint.
            self.register_buffer(_PACKED_INPUT_PROJECTION_NAME, None, persistent=False)
            self._ascend_input_projection_ready = False
            self._ascend_input_projection_initialized = False
            self._ascend_input_projection_spec = None

    def get_attn_backend(self) -> type[AttentionBackend]:
        return AscendFlashKDAAttentionBackend if FLASH_KDA_ENABLED else AscendGDNAttentionBackend

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        del positions
        num_tokens = hidden_states.size(0)
        # Token count is symbolic under vLLM's broad compile range. A Python
        # shape threshold would be fixed by the first trace and reused for
        # other batch sizes; select one mathematical path by config only.
        if getattr(self, "_ascend_fuse_input_projections", False):
            qkv, f_values, g_values, beta_logits = self._run_fused_input_projection(hidden_states)
            q, k, v = qkv.split(self.projection_size_per_partition, dim=-1)
            beta = (beta_logits if FLASH_KDA_ENABLED else beta_logits.float().sigmoid()).unsqueeze(0)
            raw_gate = rearrange(f_values, "n (h d) -> 1 n h d", d=self.head_dim)
            output_gate = rearrange(g_values, "n (h d) -> n h d", d=self.head_dim)
        else:
            if self.separate_b_proj:
                assert self.qkv_proj is not None and self.b_proj is not None
                qkv = self.qkv_proj(hidden_states)[0]
                q, k, v = qkv.split(self.projection_size_per_partition, dim=-1)
                beta_logits = self.b_proj(hidden_states)[0]
            else:
                assert self.qkvb_proj is not None
                qkvb = self.qkvb_proj(hidden_states)[0]
                q, k, v, beta_logits = qkvb.split(
                    [
                        self.projection_size_per_partition,
                        self.projection_size_per_partition,
                        self.projection_size_per_partition,
                        self.local_num_heads,
                    ],
                    dim=-1,
                )

            # FlashKDA fuses sigmoid and must receive the original BF16 logits.
            # Never reconstruct them with logit(sigmoid(beta)), which loses tails.
            beta = (beta_logits if FLASH_KDA_ENABLED else beta_logits.float().sigmoid()).unsqueeze(0)
            # AscendC performs the A_log/dt_bias gate conversion in-kernel. Passing
            # vLLM's CUDA-preprocessed gate here would apply that conversion twice.
            raw_gate = rearrange(self.f_proj(hidden_states)[0], "n (h d) -> 1 n h d", d=self.head_dim)
            output_gate = rearrange(
                self.g_proj(hidden_states)[0],
                "n (h d) -> n h d",
                d=self.head_dim,
            )
        core_attn_out = torch.empty(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops.vllm.bailing_v3_kda_attention(q, k, v, raw_gate, beta, core_attn_out, self.prefix)
        # Match Kimi's registered norm path while retaining its parameters.
        self.o_norm.forward_oot(core_attn_out, output_gate, out=core_attn_out)
        core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
        output[:] = self.o_proj(core_attn_out)[0]

    @torch.no_grad()
    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        # AttentionLayerBase's deferred post-load runs after all three child
        # convs. During RL reload it also restores this parent's original
        # storage before packing, regardless of the checkpoint loading order.
        self._ascend_conv_weights_ready = False
        if getattr(self, "_ascend_fuse_input_projections", False):
            self._ascend_input_projection_ready = False
        parent_postload = getattr(super(), "process_weights_after_loading", None)
        if parent_postload is not None:
            parent_postload(act_dtype)
        packed = self.get_buffer(_PACKED_CONV_WEIGHT_NAME)
        if packed.is_meta or packed.dtype != act_dtype:
            raise RuntimeError("Bailing KDA packed conv storage must be materialized in the model activation dtype.")
        weights = [conv.weight for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d)]
        if any(weight.is_meta for weight in weights):
            raise RuntimeError("Bailing KDA conv weights must all be loaded before packing.")
        expected_shape = (packed.shape[1] // 3, 1, packed.shape[0])
        if any(weight.shape != expected_shape for weight in weights):
            raise RuntimeError("Bailing KDA conv weight shapes cannot change after initialization.")
        if packed.device != weights[0].device:
            # device_loading_context moves CPU parameters, but not buffers.
            # Follow the loaded children once, before a graph can capture this
            # storage. Later reloads must preserve the original device/address.
            if self._ascend_conv_weights_initialized:
                raise RuntimeError("Bailing KDA packed conv weights cannot change device after initialization.")
            packed = torch.empty_like(packed, device=weights[0].device)
            self.register_buffer(_PACKED_CONV_WEIGHT_NAME, packed, persistent=False)
        packed_weight = pack_kda_conv_weights(weights, dtype=act_dtype, device=packed.device)
        packed.copy_(packed_weight)
        self._ascend_conv_weights_initialized = True
        self._ascend_conv_weights_ready = True
        if getattr(self, "_ascend_fuse_input_projections", False):
            self._pack_input_projection_weights()

    def _validate_input_projection_fusion(self) -> None:
        vllm_config = get_current_vllm_config_or_none()
        if getattr(vllm_config, "lora_config", None) is not None:
            # The merged public GEMM cannot apply LoRA wrapper updates.
            raise ValueError("Fused KDA input projections do not support LoRA")
        transfer_config = getattr(vllm_config, "weight_transfer_config", None)
        if getattr(transfer_config, "backend", None) == "sparse_nccl":
            # Sparse updates patch canonical parameters in place without the
            # deferred post-load hook that must refresh this derived buffer.
            raise ValueError("Fused KDA input projections do not support sparse_nccl weight updates")
        if self.num_speculative_tokens or self.separate_b_proj:
            raise ValueError("Fused KDA input projections require no MTP and a combined QKVB projection")
        if self.projection_size_per_partition % _KDA_INPUT_PROJECTION_ALIGNMENT:
            raise ValueError("Fused KDA input projections require a 16-aligned local projection width")
        if getattr(self.model_config, "dtype", torch.bfloat16) != torch.bfloat16:
            raise ValueError("Fused KDA input projections require a BF16 model dtype")
        layers = (self.qkvb_proj, self.f_proj, self.g_proj)
        for layer in layers:
            if (
                layer is None
                or not isinstance(layer.quant_method, UnquantizedLinearMethod)
                or layer.weight.dtype != torch.bfloat16
                or getattr(layer, "bias", None) is not None
                or getattr(layer, "gather_output", False)
                or getattr(layer, "custom_op", None) is not None
            ):
                raise ValueError("Fused KDA input projections require local bias-free BF16 unquantized linears")

    @torch.no_grad()
    def _pack_input_projection_weights(self) -> None:
        self._ascend_input_projection_ready = False
        self._validate_input_projection_fusion()
        projection = self.projection_size_per_partition
        heads = self.local_num_heads
        canonical = (self.qkvb_proj.weight, self.f_proj.weight, self.g_proj.weight)
        expected_shapes = (
            (3 * projection + heads, self.hidden_size),
            (projection, self.hidden_size),
            (projection, self.hidden_size),
        )
        if any(weight.is_meta or tuple(weight.shape) != shape for weight, shape in zip(canonical, expected_shapes)):
            raise RuntimeError("Fused KDA projection weights must be materialized with their original shapes")
        if any(weight.device != canonical[0].device for weight in canonical):
            raise RuntimeError("Fused KDA projection weights must share one device")
        qkvb, f_weight, g_weight = (_input_projection_weight_nd(weight) for weight in canonical)
        padding = (-(5 * projection + heads)) % _KDA_INPUT_PROJECTION_ALIGNMENT
        # Put beta after the wider gates so QKV/F/G/beta each start on a
        # 32-byte boundary. No checkpoint values or activation order change.
        packed = torch.cat(
            (
                qkvb[: 3 * projection],
                f_weight,
                g_weight,
                qkvb[3 * projection :],
                qkvb.new_zeros((padding, self.hidden_size)),
            ),
            dim=0,
        )
        if packed.device.type == "npu":
            packed = maybe_trans_nz(packed)
        spec = (
            tuple(packed.shape),
            packed.device,
            packed.dtype,
            _input_projection_weight_format(packed),
        )
        previous = self.get_buffer(_PACKED_INPUT_PROJECTION_NAME)
        if not self._ascend_input_projection_initialized:
            # Deferred post-load runs on the canonical parameters' real device.
            self.register_buffer(_PACKED_INPUT_PROJECTION_NAME, packed, persistent=False)
            self._ascend_input_projection_spec = spec
            self._ascend_input_projection_initialized = True
        else:
            previous_spec = (
                None
                if previous is None
                else (
                    tuple(previous.shape),
                    previous.device,
                    previous.dtype,
                    _input_projection_weight_format(previous),
                )
            )
            if previous_spec != self._ascend_input_projection_spec or spec != self._ascend_input_projection_spec:
                raise RuntimeError(
                    "Fused KDA projection shape, dtype, device and format cannot change after initialization"
                )
            previous.copy_(packed)
        self._ascend_input_projection_ready = True

    def _run_fused_input_projection(self, hidden_states: torch.Tensor):
        if not self._ascend_input_projection_ready:
            raise RuntimeError("Fused KDA projection weights require deferred post-load/refit processing")
        if hidden_states.dtype != torch.bfloat16:
            raise ValueError("Fused KDA input projections require BF16 hidden states")
        projection = self.projection_size_per_partition
        heads = self.local_num_heads
        padding = (-(5 * projection + heads)) % _KDA_INPUT_PROJECTION_ALIGNMENT
        packed = self.get_buffer(_PACKED_INPUT_PROJECTION_NAME)
        projected = torch.ops.vllm.unquantized_gemm(hidden_states, packed, None)
        qkv, f_values, g_values, beta, _padding = projected.split(
            (3 * projection, projection, projection, heads, padding), dim=-1
        )
        return qkv, f_values, g_values, beta

    def _conv_weights_t(self, dtype: torch.dtype) -> torch.Tensor:
        packed = self.get_buffer(_PACKED_CONV_WEIGHT_NAME)
        if not self._ascend_conv_weights_ready or packed.is_meta:
            raise RuntimeError("Bailing KDA conv weights must be prepared after weight loading before forward.")
        if packed.dtype != dtype:
            raise RuntimeError("Bailing KDA conv weights and activations must have the same dtype.")
        return packed

    def _run_causal_conv1d(
        self,
        mixed_qkv: torch.Tensor,
        conv_weights_t: torch.Tensor,
        conv_state_storage: torch.Tensor,
        query_start_loc: torch.Tensor,
        cache_indices: torch.Tensor,
        initial_state_mode: torch.Tensor | None,
        *,
        run_mode: int,
        max_query_len: int = -1,
        num_accepted_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if run_mode not in (0, 1):
            raise ValueError(f"Unsupported causal_conv1d run_mode: {run_mode}")
        # Convolution uses one history buffer per request, including MTP.
        if cache_indices.dim() == 2:
            cache_indices = cache_indices[:, 0]
        elif cache_indices.dim() != 1:
            raise ValueError(f"KDA causal cache_indices must be 1D or 2D, got shape={tuple(cache_indices.shape)}")
        cache_indices = cache_indices.contiguous()
        if cache_indices.shape[0] == 0:
            return torch.zeros_like(mixed_qkv)

        conv_state = conv_state_storage.transpose(-1, -2) if self._conv_state_dim_first else conv_state_storage
        kernel_state = conv_state
        kernel_indices = cache_indices
        null_block_id = NULL_BLOCK_ID
        copy_state_back = not native_conv_state_supported(conv_state, mixed_qkv.dtype)
        state_len, state_dim = conv_state.shape[1:]
        sd_layout = conv_state.stride(2) == 1 and conv_state.stride(1) >= state_dim
        ds_layout = conv_state.stride(1) == 1 and conv_state.stride(2) >= state_len
        row_span = (state_len - 1) * conv_state.stride(1) + (state_dim - 1) * conv_state.stride(2) + 1
        update_state_during_gather = (
            copy_state_back
            and run_mode == 1
            and max_query_len == 1
            and num_accepted_tokens is None
            and self.num_speculative_tokens == 0
            and conv_weights_t.shape[0] == 4
            and 3 <= conv_state.shape[1] <= 32
            # FLA switches to compact varlen addressing when these differ.
            and mixed_qkv.ndim == 2
            and mixed_qkv.shape[0] == cache_indices.shape[0]
            and mixed_qkv.dtype == torch.bfloat16
            and mixed_qkv.stride(1) == 1
            and conv_state.dtype in (torch.float32, torch.bfloat16)
            and (sd_layout or ds_layout)
            and conv_state.stride(0) >= row_span
        )
        if copy_state_back:
            # Pack only this batch's rows. The copy kernel handles DS/page
            # strides and casts on load/store without copying the cache pool.
            kernel_state = torch.empty(
                (cache_indices.shape[0], *conv_state.shape[1:]),
                dtype=mixed_qkv.dtype,
                device=conv_state.device,
            )
            kernel_indices = torch.empty_like(cache_indices, dtype=torch.int32)
            # Packed row zero is valid. FLA only enables null-block filtering
            # for nonnegative IDs, so reserve the ID just past the packed rows.
            null_block_id = kernel_state.shape[0]
            if update_state_during_gather:
                # Keep native convolution arithmetic, but commit its single-token
                # history update while gathering the old BF16 state. No scatter
                # is needed after the native op mutates its temporary packed rows.
                gather_update_conv_state(
                    conv_state,
                    kernel_state,
                    mixed_qkv,
                    cache_indices,
                    query_start_loc,
                    kernel_indices,
                    null_block_id=NULL_BLOCK_ID,
                )
            else:
                cache_indices = torch.where(cache_indices == NULL_BLOCK_ID, PAD_SLOT_ID, cache_indices)
                copy_conv_state(
                    conv_state, kernel_state, cache_indices, query_start_loc, kernel_indices, write_back=False
                )

        conv_output = None
        fuse_native_decode_prepare = (
            run_mode == 1
            and max_query_len == 1
            and num_accepted_tokens is None
            and mixed_qkv.ndim == 2
            and _native_decode_prepare_compatible(
                mixed_qkv.shape[0],
                mixed_qkv.dtype,
                conv_state,
                cache_indices,
                query_start_loc,
                self.num_speculative_tokens,
            )
        )
        if fuse_native_decode_prepare:
            if mixed_qkv.is_contiguous():
                # CC1 and genuinely packed projections retain the original
                # prepare kernel, with no projection copy.
                conv_output = torch.empty_like(mixed_qkv)
                kernel_indices = prepare_native_conv_decode(
                    cache_indices, query_start_loc, conv_state.shape[0], null_block_id, conv_output
                )
            else:
                packed_input = torch.empty(mixed_qkv.shape, dtype=mixed_qkv.dtype, device=mixed_qkv.device)
                conv_output = torch.empty_like(packed_input)
                kernel_indices = prepare_native_conv_decode(
                    cache_indices,
                    query_start_loc,
                    conv_state.shape[0],
                    null_block_id,
                    conv_output,
                    input_tensor=mixed_qkv,
                    packed_input=packed_input,
                )
                mixed_qkv = packed_input
        elif not copy_state_back:
            # Native fixed-row decode does not check query lengths or cache
            # bounds. Keep the staging path's inactive-row semantics when
            # passing a same-dtype SD cache (including physical page gaps).
            active = (
                (cache_indices > NULL_BLOCK_ID)
                & (cache_indices < conv_state.shape[0])
                & (query_start_loc[1:] > query_start_loc[:-1])
            )
            kernel_indices = torch.where(active, cache_indices, null_block_id)
        elif not update_state_during_gather:
            # FLA update has no pad_slot_id argument.
            kernel_indices = torch.where(kernel_indices == PAD_SLOT_ID, null_block_id, kernel_indices)

        if run_mode == 1 and conv_output is None:
            conv_output = torch.zeros_like(mixed_qkv)
        output = run_causal_conv1d(
            mixed_qkv,
            conv_weights_t,
            kernel_state,
            query_start_loc,
            kernel_indices,
            initial_state_mode,
            run_mode=run_mode,
            max_query_len=max_query_len,
            num_accepted_tokens=num_accepted_tokens,
            null_block_id=null_block_id,
            out=conv_output,
        )

        if copy_state_back and not update_state_during_gather:
            copy_conv_state(conv_state, kernel_state, cache_indices, query_start_loc, kernel_indices, write_back=True)
        return output

    def _run_recurrent(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        raw_gate: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        state_indices: torch.Tensor,
        *,
        num_accepted_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if FLASH_KDA_ENABLED:
            # Keep the established AscendC recurrent/decode arithmetic even
            # when prefill uses raw logits for FlashKDA (including mixed/MTP).
            beta = beta.float().sigmoid()
        if num_accepted_tokens is not None:
            # Accepted count describes the previous verification, whose length
            # may exceed this query after a shortened speculative proposal.
            num_accepted_tokens = rebase_short_kda_queries(
                recurrent_state, state_indices, cu_seqlens, num_accepted_tokens
            )
        return run_recurrent_kda(
            q,
            k,
            v,
            raw_gate,
            beta,
            recurrent_state,
            cu_seqlens,
            state_indices,
            self.A_log,
            self.dt_bias,
            lower_bound=self.lower_bound if self.safe_gate else None,
            num_accepted_tokens=num_accepted_tokens,
        )

    def _run_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        raw_gate: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_state: torch.Tensor,
        prebuilt_metadata,
    ) -> torch.Tensor:
        cu_seqlens = (
            prebuilt_metadata.cu_seqlens_host
            if prebuilt_metadata.cu_seqlens_kern is None
            else prebuilt_metadata.cu_seqlens_kern
        )
        if has_initial_state is None:
            raise ValueError("KDA prefill requires has_initial_state metadata")
        keep = prebuilt_metadata.keep_meta
        if keep is not None:
            state_indices = state_indices[keep]
            has_initial_state = has_initial_state[keep]

        # Hybrid pages can pad the state row stride on A3 as well as A5.
        # The worker selects the shared fused or byte-copy plan; both gather
        # only selected rows without materializing the whole state pool.
        state_copy = getattr(self, "_ascend_kda_state_copy", None)
        if state_copy is None or not getattr(self, "_kda_state_copy_ready", False):
            raise RuntimeError("KDA state copy requires worker cache initialization before prefill")
        initial_state = state_copy.gather(recurrent_state, state_indices, has_initial_state)
        prefill_kwargs = {"lower_bound": self.lower_bound if self.safe_gate else None}
        if FLASH_KDA_ENABLED:
            if prebuilt_metadata.flash_cu_seqlens is None:
                raise RuntimeError("FlashKDA prefill requires batch device cu_seqlens metadata")
            prefill_kwargs["cu_seqlens_device"] = prebuilt_metadata.flash_cu_seqlens
        output, final_state = run_chunk_kda(
            q,
            k,
            v,
            raw_gate,
            beta,
            initial_state,
            cu_seqlens,
            prebuilt_metadata.chunk_indices_chunk64_host,
            self.A_log,
            self.dt_bias,
            **prefill_kwargs,
        )
        state_copy.scatter(recurrent_state, final_state, state_indices)
        return output

    @eager_break_during_capture
    def _forward(
        self,
        q_proj_states: torch.Tensor,
        k_proj_states: torch.Tensor,
        v_proj_states: torch.Tensor,
        g1: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        attn_metadata_map = get_forward_context().attn_metadata
        if attn_metadata_map is None:
            core_attn_out.zero_()
            return

        assert isinstance(attn_metadata_map, dict)
        attn_metadata = attn_metadata_map.get(self.prefix)
        if attn_metadata is None:
            core_attn_out.zero_()
            return
        assert isinstance(attn_metadata, GDNAttentionMetadata)

        num_actual_tokens = attn_metadata.num_actual_tokens
        conv_state, recurrent_state = self.kv_cache
        spec_masks = attn_metadata.spec_sequence_masks
        defer_qkv_pack = False
        if (
            spec_masks is None
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes > 0
            and q_proj_states.dtype == torch.bfloat16
        ):
            decode_meta = attn_metadata.non_spec_decode_metadata
            if decode_meta is not None:
                native_state = conv_state.transpose(-1, -2) if self._conv_state_dim_first else conv_state
                defer_qkv_pack = _native_decode_prepare_compatible(
                    num_actual_tokens,
                    q_proj_states.dtype,
                    native_state,
                    decode_meta.causal_conv1d.cache_indices,
                    decode_meta.causal_conv1d.query_start_loc,
                    self.num_speculative_tokens,
                )
        mixed_qkv = _contiguous_packed_qkv(
            q_proj_states[:num_actual_tokens],
            k_proj_states[:num_actual_tokens],
            v_proj_states[:num_actual_tokens],
            allow_row_gaps=defer_qkv_pack,
        )
        # Keep the upstream custom-op keyword name (g1), while treating the
        # tensor as the raw gate required by the AscendC kernels.
        raw_gate = g1[:, :num_actual_tokens]
        beta = beta[:, :num_actual_tokens]

        conv_weights_t = self._conv_weights_t(mixed_qkv.dtype)
        spec_token_indices = attn_metadata.spec_token_indx
        non_spec_token_indices = attn_metadata.non_spec_token_indx

        if spec_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_spec = mixed_qkv
                gate_spec = raw_gate
                beta_spec = beta
                mixed_non_spec = gate_non_spec = beta_non_spec = None
            else:
                assert spec_token_indices is not None and non_spec_token_indices is not None
                mixed_spec = mixed_qkv.index_select(0, spec_token_indices)
                gate_spec = raw_gate.index_select(1, spec_token_indices)
                beta_spec = beta.index_select(1, spec_token_indices)
                mixed_non_spec = mixed_qkv.index_select(0, non_spec_token_indices)
                gate_non_spec = raw_gate.index_select(1, non_spec_token_indices)
                beta_non_spec = beta.index_select(1, non_spec_token_indices)
        else:
            mixed_spec = gate_spec = beta_spec = None
            mixed_non_spec = mixed_qkv
            gate_non_spec = raw_gate
            beta_non_spec = beta

        core_spec = None
        if mixed_spec is not None:
            spec_meta = attn_metadata.spec_decode_metadata
            assert spec_meta is not None
            conv_meta = spec_meta.spec_causal_conv1d
            mixed_spec = self._run_causal_conv1d(
                mixed_spec,
                conv_weights_t,
                conv_state,
                conv_meta.query_start_loc,
                conv_meta.cache_indices,
                None,
                run_mode=1,
                max_query_len=self.num_speculative_tokens + 1,
                num_accepted_tokens=conv_meta.num_accepted_tokens,
            )
            q_spec, k_spec, v_spec = (
                rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim) for x in mixed_spec.chunk(3, dim=-1)
            )
            assert gate_spec is not None and beta_spec is not None
            assert attn_metadata.spec_query_start_loc is not None
            assert attn_metadata.spec_state_indices_tensor is not None
            core_spec = self._run_recurrent(
                q_spec,
                k_spec,
                v_spec,
                gate_spec,
                beta_spec,
                recurrent_state,
                attn_metadata.spec_query_start_loc,
                attn_metadata.spec_state_indices_tensor,
                num_accepted_tokens=conv_meta.num_accepted_tokens,
            )
            core_spec = _zero_padded_spec_output(core_spec, attn_metadata.spec_query_start_loc)

        core_non_spec = None
        if mixed_non_spec is not None and mixed_non_spec.shape[0] > 0:
            if attn_metadata.num_prefills > 0:
                prefill_meta = attn_metadata.non_spec_prefill_metadata
                assert prefill_meta is not None
                conv_meta = prefill_meta.causal_conv1d
                mixed_non_spec = self._run_causal_conv1d(
                    mixed_non_spec,
                    conv_weights_t,
                    conv_state,
                    conv_meta.query_start_loc,
                    conv_meta.cache_indices,
                    conv_meta.initial_state_mode,
                    run_mode=0,
                )
            elif attn_metadata.num_decodes > 0:
                decode_meta = attn_metadata.non_spec_decode_metadata
                assert decode_meta is not None
                conv_meta = decode_meta.causal_conv1d
                mixed_non_spec = self._run_causal_conv1d(
                    mixed_non_spec,
                    conv_weights_t,
                    conv_state,
                    conv_meta.query_start_loc,
                    conv_meta.cache_indices,
                    None,
                    run_mode=1,
                    max_query_len=1,
                )

            q_non_spec, k_non_spec, v_non_spec = (
                rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim) for x in mixed_non_spec.chunk(3, dim=-1)
            )
            assert gate_non_spec is not None and beta_non_spec is not None

            split_non_spec = spec_masks is None and attn_metadata.num_prefills > 0 and attn_metadata.num_decodes > 0
            num_decode_tokens = attn_metadata.num_decode_tokens
            core_decode = None
            if split_non_spec:
                assert attn_metadata.non_spec_query_start_loc is not None
                assert attn_metadata.non_spec_state_indices_tensor is not None
                core_decode = self._run_recurrent(
                    q_non_spec[:, :num_decode_tokens],
                    k_non_spec[:, :num_decode_tokens],
                    v_non_spec[:, :num_decode_tokens],
                    gate_non_spec[:, :num_decode_tokens],
                    beta_non_spec[:, :num_decode_tokens],
                    recurrent_state,
                    attn_metadata.non_spec_query_start_loc[: attn_metadata.num_decodes + 1],
                    attn_metadata.non_spec_state_indices_tensor[: attn_metadata.num_decodes],
                )

            if attn_metadata.num_prefills > 0:
                if split_non_spec:
                    q_non_spec = q_non_spec[:, num_decode_tokens:]
                    k_non_spec = k_non_spec[:, num_decode_tokens:]
                    v_non_spec = v_non_spec[:, num_decode_tokens:]
                    gate_non_spec = gate_non_spec[:, num_decode_tokens:]
                    beta_non_spec = beta_non_spec[:, num_decode_tokens:]
                assert attn_metadata.prefill_state_indices is not None
                assert attn_metadata.prefill_has_initial_state is not None
                prefill_meta = attn_metadata.non_spec_prefill_metadata
                assert prefill_meta is not None
                core_prefill = self._run_prefill(
                    q_non_spec,
                    k_non_spec,
                    v_non_spec,
                    gate_non_spec,
                    beta_non_spec,
                    recurrent_state,
                    attn_metadata.prefill_state_indices,
                    attn_metadata.prefill_has_initial_state,
                    prefill_meta.chunk,
                )
                core_non_spec = (
                    torch.cat((core_decode, core_prefill), dim=1) if core_decode is not None else core_prefill
                )
            elif attn_metadata.num_decodes > 0:
                assert attn_metadata.non_spec_query_start_loc is not None
                assert attn_metadata.non_spec_state_indices_tensor is not None
                core_non_spec = self._run_recurrent(
                    q_non_spec,
                    k_non_spec,
                    v_non_spec,
                    gate_non_spec,
                    beta_non_spec,
                    recurrent_state,
                    attn_metadata.non_spec_query_start_loc[: attn_metadata.num_decodes + 1],
                    attn_metadata.non_spec_state_indices_tensor,
                )

        if core_spec is None and core_non_spec is None:
            core_attn_out.zero_()
            return

        if core_spec is not None and core_non_spec is not None:
            # The two indexed writes can leave holes in graph-shaped metadata.
            core_attn_out[:, :num_actual_tokens].zero_()
            assert spec_token_indices is not None and non_spec_token_indices is not None
            core_attn_out[:, :num_actual_tokens].index_copy_(1, spec_token_indices, core_spec)
            core_attn_out[:, :num_actual_tokens].index_copy_(1, non_spec_token_indices, core_non_spec)
        elif core_spec is not None:
            core_attn_out[:, :num_actual_tokens] = core_spec
        elif core_non_spec is not None:
            core_attn_out[:, :num_actual_tokens] = core_non_spec
        core_attn_out[:, num_actual_tokens:].zero_()
