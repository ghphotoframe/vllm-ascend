# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""run_chunk_kda must use the dense kernel path for a single full-length sequence."""

from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ops.kda import run_chunk_kda


def _call(cu_seqlens, chunk_indices, num_tokens=8192):
    q = torch.zeros(1, num_tokens, 4, 128, dtype=torch.bfloat16)
    seen = {}

    def fake_chunk_kda_fwd(q_arg, k_arg, v_arg, g_arg, beta_arg, *_args, **kwargs):
        seen.update(kwargs)
        return q_arg, beta_arg, None

    with (
        patch("vllm_ascend.ops.kda.chunk_kda_fwd", side_effect=fake_chunk_kda_fwd),
        patch("vllm.third_party.flash_linear_attention.ops.l2norm.l2norm_fwd", side_effect=lambda x: x),
    ):
        run_chunk_kda(
            q,
            q,
            q,
            q,
            torch.zeros(1, num_tokens, 4, dtype=torch.float32),
            torch.zeros(1, 4, 128, 128, dtype=torch.float32),
            cu_seqlens,
            chunk_indices,
            torch.zeros(4, dtype=torch.float32),
            torch.zeros(4 * 128, dtype=torch.float32),
            lower_bound=-5.0,
        )
    return seen


def test_single_full_sequence_uses_dense_path():
    seen = _call((0, 8192), (0, 0, 64, 64))
    assert seen["cu_seqlens"] is None
    assert seen["chunk_indices"] is None


@pytest.mark.parametrize(
    "cu_seqlens",
    [
        (0, 4096),  # single sequence shorter than the padded token block
        (0, 4096, 8192),  # two sequences
        (0, 1024, 2048, 8192),  # three sequences
    ],
)
def test_multi_or_padded_sequence_keeps_varlen_path(cu_seqlens):
    chunk_indices = tuple(range(4))
    seen = _call(cu_seqlens, chunk_indices)
    assert seen["cu_seqlens"] == cu_seqlens
    assert seen["chunk_indices"] == chunk_indices


def test_missing_segment_metadata_stays_dense():
    seen = _call(None, None)
    assert seen["cu_seqlens"] is None
    assert seen["chunk_indices"] is None
