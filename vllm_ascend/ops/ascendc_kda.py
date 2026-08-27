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

from typing import Any

import torch

from vllm_ascend.utils import (
    AscendDeviceType,
    enable_custom_op,
    get_ascend_device_type,
)


def _is_npu_tensor(tensor: torch.Tensor) -> bool:
    return tensor.device.type == "npu"



def _ascendc_recurrent_kda_runtime_available() -> bool:
    if get_ascend_device_type() not in {
        AscendDeviceType.A2,
        AscendDeviceType.A3,
    }:
        return False
    return enable_custom_op() and hasattr(torch.ops._C_ascend, "npu_recurrent_kda")


def can_use_ascendc_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor | None = None,
) -> bool:
    """Return whether the migrated recurrent KDA contract is satisfied."""
    if not _ascendc_recurrent_kda_runtime_available():
        return False

    tensors = (
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        cu_seqlens,
        ssm_state_indices,
    )
    if not all(_is_npu_tensor(tensor) for tensor in tensors):
        return False
    if any(tensor.device != q.device for tensor in tensors[1:]):
        return False
    if num_accepted_tokens is not None and (
        not _is_npu_tensor(num_accepted_tokens) or num_accepted_tokens.device != q.device
    ):
        return False

    if q.dim() not in (3, 4) or k.shape != q.shape:
        return False
    if q.dim() == 3:
        total_tokens, num_heads, key_dim = q.shape
        if v.dim() != 3 or g.dim() != 3 or beta.dim() != 2:
            return False
        value_tokens, num_value_heads, value_dim = v.shape
        expected_gate_shape = (total_tokens, num_value_heads, key_dim)
        expected_beta_shape = (total_tokens, num_value_heads)
    else:
        batch, sequence_length, num_heads, key_dim = q.shape
        if batch != 1 or v.dim() != 4 or g.dim() != 4 or beta.dim() != 3:
            return False
        total_tokens = batch * sequence_length
        value_batch, value_sequence_length, num_value_heads, value_dim = v.shape
        if value_batch != batch or value_sequence_length != sequence_length:
            return False
        value_tokens = total_tokens
        expected_gate_shape = (
            batch,
            sequence_length,
            num_value_heads,
            key_dim,
        )
        expected_beta_shape = (batch, sequence_length, num_value_heads)

    if (
        total_tokens <= 0
        or value_tokens != total_tokens
        or g.shape != expected_gate_shape
        or beta.shape != expected_beta_shape
        or num_heads <= 0
        or num_value_heads <= 0
        or num_value_heads % num_heads != 0
        or key_dim != 128
        or value_dim != 128
    ):
        return False
    if (
        q.dtype != torch.bfloat16
        or k.dtype != torch.bfloat16
        or v.dtype != torch.bfloat16
        or g.dtype not in {torch.float32, torch.bfloat16, torch.float16}
        or beta.dtype not in {torch.float32, torch.bfloat16, torch.float16}
    ):
        return False

    if (
        initial_state.dim() != 4
        or initial_state.shape[0] < 1
        or initial_state.shape[1:] != (num_value_heads, value_dim, key_dim)
        or initial_state.dtype not in {torch.float32, torch.bfloat16}
    ):
        return False
    if cu_seqlens.dim() != 1 or cu_seqlens.numel() < 2 or cu_seqlens.dtype not in {torch.int32, torch.int64}:
        return False

    sequence_count = cu_seqlens.numel() - 1
    packed_indices = ssm_state_indices.dim() == 1 and ssm_state_indices.numel() >= total_tokens
    speculative_indices = (
        ssm_state_indices.dim() == 2
        and ssm_state_indices.shape[0] == sequence_count
        and 0 < ssm_state_indices.shape[1] <= 8
    )
    if ssm_state_indices.dtype not in {torch.int32, torch.int64} or not (packed_indices or speculative_indices):
        return False
    return not (
        num_accepted_tokens is not None
        and (
            num_accepted_tokens.dim() != 1
            or num_accepted_tokens.shape[0] != sequence_count
            or num_accepted_tokens.dtype not in {torch.int32, torch.int64}
        )
    )


def ascendc_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor | None = None,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run recurrent KDA using an already precomputed step log gate."""
    if scale is None:
        scale = k.shape[-1] ** -0.5
    output = torch.ops._C_ascend.npu_recurrent_kda(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.contiguous(),
        beta.contiguous(),
        initial_state,
        cu_seqlens,
        ssm_state_indices,
        A_log=None,
        dt_bias=None,
        num_accepted_tokens=num_accepted_tokens,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
        allow_neg_eigval=False,
        safe_gate=False,
        lower_bound=-5.0,
    )
    return output, initial_state


def recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    scale: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run Bailing decode using the AscendC KDA operators."""
    return ascendc_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
