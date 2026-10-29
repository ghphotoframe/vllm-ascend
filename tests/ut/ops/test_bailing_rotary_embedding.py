# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

from vllm_ascend.ops.bailing_rotary_embedding import AscendBailingMRotaryEmbedding


def _make_embedding(dtype=torch.float32):
    embedding = AscendBailingMRotaryEmbedding.__new__(AscendBailingMRotaryEmbedding)
    torch.nn.Module.__init__(embedding)
    embedding.head_size = 80
    embedding.rotary_dim = 64
    embedding.spatial_size = 24
    # Real checkpoint sections: temporal 8, height 12, width 12.
    angles = torch.arange(256, dtype=torch.float32)[:, None] * (
        1 / 6000000 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64)
    )
    embedding.cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(dtype)
    embedding.apply_rotary_emb = lambda x, cos, sin: ApplyRotaryEmb.forward_static(x, cos, sin, False)
    return embedding


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("three_axes", [False, True])
def test_bailing_mla_coefficients_signed_axes_and_stable_cache(dtype, three_axes):
    embedding = _make_embedding(dtype)
    positions = torch.tensor([[7, 8, 9], [-2, 0, 2], [3, -1, -3]])
    if not three_axes:
        positions = positions[0]
    cache = (torch.full((8, 1, 1, 64), 42, dtype=dtype), torch.full((8, 1, 1, 64), 42, dtype=dtype))
    pointers = tuple(t.data_ptr() for t in cache)
    for current in (positions, -positions, torch.zeros_like(positions)):
        cos, sin = embedding.select_cos_sin_mla(current, cache)
        expected_cos, expected_sin = embedding.select_cos_sin(current)
        torch.testing.assert_close(cos[:, 0, 0], expected_cos.repeat(1, 2), rtol=0, atol=0)
        torch.testing.assert_close(sin[:, 0, 0], expected_sin.repeat(1, 2), rtol=0, atol=0)
        assert (cos.data_ptr(), sin.data_ptr()) == pointers
        assert torch.all(cache[0][3:] == 42) and torch.all(cache[1][3:] == 42)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bailing_forward_shared_rotary_preserves_nope_and_strided_inputs(dtype):
    embedding = _make_embedding(dtype)
    positions = torch.tensor([[7, 8, 9], [-2, 0, 2], [3, -1, -3]])
    query_backing = torch.randn(3, 4, 96, dtype=dtype)
    key_backing = torch.randn(3, 1, 96, dtype=dtype)
    query, key = query_backing[..., :80], key_backing[..., :80]
    original_query, original_key = query_backing.clone(), key_backing.clone()
    cos, sin = embedding.select_cos_sin(positions)

    def reference(x):
        x1, x2 = x[..., :64:2], x[..., 1:64:2]
        rotated = torch.stack((x1 * cos[:, None] - x2 * sin[:, None], x2 * cos[:, None] + x1 * sin[:, None]), -1)
        return torch.cat((rotated.flatten(-2), x[..., 64:]), -1)

    actual_q, actual_k = embedding.forward_oot(positions, query, key)
    torch.testing.assert_close(actual_q, reference(query), rtol=0, atol=0)
    torch.testing.assert_close(actual_k, reference(key), rtol=0, atol=0)
    torch.testing.assert_close(query_backing, original_query, rtol=0, atol=0)
    torch.testing.assert_close(key_backing, original_key, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(2, 1, 1, 64), (8, 1, 1, 32), (8, 64)])
def test_bailing_mla_rejects_incompatible_graph_buffer(shape):
    with pytest.raises(ValueError, match="shape or capacity"):
        _make_embedding().select_cos_sin_mla(torch.arange(3), (torch.empty(shape), torch.empty(shape)))


def test_bailing_empty_and_missing_key():
    embedding = _make_embedding()
    query, key = torch.empty(0, 4, 80), torch.empty(0, 1, 80)
    result = embedding.forward_oot(torch.empty(3, 0, dtype=torch.long), query, key)
    assert result[0] is query and result[1] is key
    with pytest.raises(ValueError, match="requires key"):
        embedding.forward_oot(torch.arange(3), torch.empty(3, 80))
