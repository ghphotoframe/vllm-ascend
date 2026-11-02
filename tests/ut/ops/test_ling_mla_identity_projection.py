# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from vllm.model_executor.models.bailing_moe_v3 import _IdentityProjection

from vllm_ascend.ops.mla import _adapt_mla_output_projection


@pytest.mark.parametrize("is_prefill", [False, True])
def test_ling_identity_accepts_hint_without_changing_tuple_or_width(is_prefill):
    original = _IdentityProjection()
    adapted = _adapt_mla_output_projection(original)
    x = torch.randn(3, 896)
    y, bias = adapted(x, is_prefill=is_prefill)
    assert y is x and bias is None
    assert y.shape == (3, 896)
    assert list(adapted.parameters()) == []


@pytest.mark.parametrize("is_prefill", [False, True])
def test_normal_projection_keeps_prefill_argument(is_prefill):
    class NormalProjection(torch.nn.Module):
        def forward(self, x, *, is_prefill=False):
            return x, is_prefill

    original = NormalProjection()
    adapted = _adapt_mla_output_projection(original)
    assert adapted is original
    assert adapted(torch.zeros(1, 2), is_prefill=is_prefill)[1] == is_prefill
