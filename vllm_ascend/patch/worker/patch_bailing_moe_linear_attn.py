#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Patch for BailingMoELinear models to support Ascend NPU.

This module provides NPU-fridendly patches for the following classes:
- ``BailingMoeLinearMLA``
- ``BailingMoELinearAttention``

Patches include:

1. **BF16 Rotary Embedding**: Forces BF16 dtype for rotary embedding cache
   because NPU operators ``npu_interleave_rope`` and ``npu_kv_rmsnorm_rope_cache``
   do not support FP32 input dtype.

2. **NPU-fridendly Linear Attention Methods**: Replaces GPU-fridendly Triton kernel
   calls with NPU-fridendly implementations:

   - ``_prefill_and_mix_infer``: uses ``BailingLinearKernelNPU`` instead of
     ``MiniMaxText01LinearKernel`` 
   - ``_decode_infer``: uses ``linear_decode_forward_npu`` instead of
     ``linear_decode_forward_triton``
   - ``_forward``: fixes the group-norm branch, replacing it with
     ``current_platform``-based dispatch.
"""

import torch

# isort: off
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.utils import maybe_prefix
import vllm.model_executor.models.bailing_moe_linear as bailing_moe_linear
from vllm_ascend.ops.triton.mamba.linear_attention_npu import (
    _forward_npu,
    _prefill_and_mix_infer_npu,
)

BailingMoELinearAttention = bailing_moe_linear.BailingMoELinearAttention
BailingMoeMLAAttention = bailing_moe_linear.BailingMoeV25MLAAttention

# =============================================================================
# Patch 1: NPU-friendly Linear Attention Methods
# =============================================================================
BailingMoELinearAttention._prefill_and_mix_infer = _prefill_and_mix_infer_npu
BailingMoELinearAttention._forward = _forward_npu

# =============================================================================
# Patch 2: BF16 Rotary Embedding
# =============================================================================


def _patch_init_to_use_bf16_rope(cls):
    """Patch a class's __init__ to force BF16 dtype for rotary embedding.

    The NPU operators ``npu_interleave_rope`` and ``npu_kv_rmsnorm_rope_cache``
    used by MLA/SFA attention backends do not support FP32 input dtype.
    This patch ensures the rotary embedding cache is created with BF16 dtype.

    The module-level ``get_rope`` is temporarily replaced during ``__init__``
    and restored afterwards via ``try/finally``, so the patch is contained and
    does not permanently mutate global state.
    """
    original_init = cls.__init__
    original_get_rope = bailing_moe_linear.get_rope

    def bf16_get_rope(*args, **kwargs):
        kwargs['dtype'] = torch.bfloat16
        return original_get_rope(*args, **kwargs)

    def patched_init(self, *args, **kwargs):
        bailing_moe_linear.get_rope = bf16_get_rope
        try:
            original_init(self, *args, **kwargs)
        finally:
            bailing_moe_linear.get_rope = original_get_rope

    cls.__init__ = patched_init


# Apply BF16 rotary embedding patch to both classes
_patch_init_to_use_bf16_rope(BailingMoeMLAAttention)
_patch_init_to_use_bf16_rope(BailingMoELinearAttention)

# =============================================================================
# Patch 3: Add prefix parameter to ParallelLMHead in BailingMoeV25ForCausalLM
# =============================================================================

BailingMoeV25ForCausalLM = bailing_moe_linear.BailingMoeV25ForCausalLM


def _patch_init_to_add_lm_head_prefix(cls):
    """Patch BailingMoeV25ForCausalLM.__init__ to pass prefix to ParallelLMHead.

    The upstream ``BailingMoeV25ForCausalLM.__init__`` constructs ``lm_head``
    without a ``prefix`` argument, which causes weight-loading issues on Ascend.
    This patch temporarily replaces the module-level ``ParallelLMHead`` with a
    wrapper that injects ``prefix=maybe_prefix(prefix, "lm_head")`` automatically,
    then restores the original after ``__init__`` completes.
    """
    original_init = cls.__init__
    original_parallel_lm_head = bailing_moe_linear.ParallelLMHead

    def patched_init(self, *args, **kwargs):
        # Extract prefix from kwargs (default "") to build the lm_head prefix
        _prefix = kwargs.get("prefix", "")

        class _ParallelLMHeadWithPrefix(original_parallel_lm_head):
            def __init__(inner_self, vocab_size, hidden_size, **kw):
                kw.setdefault("prefix", maybe_prefix(_prefix, "lm_head"))
                super().__init__(vocab_size, hidden_size, **kw)

        bailing_moe_linear.ParallelLMHead = _ParallelLMHeadWithPrefix
        try:
            original_init(self, *args, **kwargs)
        finally:
            bailing_moe_linear.ParallelLMHead = original_parallel_lm_head

    cls.__init__ = patched_init


# Apply prefix patch to BailingMoeV25ForCausalLM
_patch_init_to_add_lm_head_prefix(BailingMoeV25ForCausalLM)