# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
import torch_npu

from vllm_ascend.ops.mm_encoder_attention import AscendMMEncoderAttention, _npu_format_code


@pytest.mark.parametrize("tokens", [4056, 8112, 32768])
@pytest.mark.parametrize("adjacent", [True, False])
def test_shared_qk_pad_matches_serving_layout(tokens, adjacent):
    raw = torch.randn(2, tokens, 2, 72, dtype=torch.bfloat16, device="npu")
    q, k = torch_npu.npu_format_cast(raw, 0).unbind(dim=0)
    v_storage = torch.randn(tokens, 3, 144, dtype=torch.bfloat16, device="npu")
    v = v_storage[:, 2].view(tokens, 2, 72)
    assert tuple(_npu_format_code(t) for t in (q, k, v)) == (0, 0, 2)
    assert k.storage_offset() == tokens * 144 and v.storage_offset() == 288
    assert q.stride() == k.stride() == (144, 72, 1) and v.stride() == (432, 72, 1)

    # Reverse the pair to exercise fallback even with equal storage addresses.
    if not adjacent:
        q, k = k, q
    layer = SimpleNamespace(enable_pad=True, _fuse_vit_qk_padding=True)
    expected = [F.pad(t, (0, 56)).cpu().view(torch.int16) for t in (q, k, v)]

    with patch.object(F, "pad", wraps=F.pad) as pad:
        result = AscendMMEncoderAttention._maybe_pad_qkv(layer, q, k, v, allow_shared_qk=True)
    assert pad.call_count == (2 if adjacent else 3)
    assert result[-1] == 72
    for actual, reference in zip(result[:3], expected):
        assert actual.is_contiguous()
        assert torch.equal(actual.cpu().view(torch.int16), reference)
