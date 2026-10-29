# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.ops.kda_conv import pack_kda_conv_weights


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strided", [False, True])
def test_combined_and_separate_weights_have_identical_fla_layout(dtype, strided):
    source = torch.arange(3 * 6 * 8, dtype=torch.float32).reshape(18, 1, 8)
    if strided:
        source = source[..., ::2]
        assert not source.is_contiguous()
    before = source.clone()
    q, k, v = source.chunk(3, dim=0)

    combined = pack_kda_conv_weights((source,), dtype=dtype, device="cpu")
    separate = pack_kda_conv_weights((q, k, v), dtype=dtype, device=torch.device("cpu"))

    expected = source[:, 0, :].transpose(0, 1).to(dtype)
    for packed in (combined, separate):
        assert packed.dtype == dtype
        assert packed.device == source.device
        assert packed.is_contiguous()
        assert packed.shape == (source.shape[-1], 18)
        torch.testing.assert_close(packed, expected, rtol=0, atol=0)
    torch.testing.assert_close(source, before, rtol=0, atol=0)


@pytest.mark.parametrize("count", [0, 2, 4])
def test_pack_rejects_incomplete_qkv(count):
    weights = (torch.empty(6, 1, 4),) * count
    with pytest.raises(ValueError, match="one combined QKV weight or three Q/K/V"):
        pack_kda_conv_weights(weights, dtype=torch.bfloat16, device="cpu")


@pytest.mark.parametrize("shape", [(6, 4), (6, 2, 4), (6, 1, 1, 4)])
def test_pack_requires_canonical_weights(shape):
    with pytest.raises(ValueError, match="shape"):
        pack_kda_conv_weights((torch.empty(shape),), dtype=torch.bfloat16, device="cpu")


@pytest.mark.parametrize("shape", [(5, 1, 4), (6, 1, 3)])
def test_pack_rejects_mismatched_qkv_shapes(shape):
    weights = (torch.empty(6, 1, 4), torch.empty(shape), torch.empty(6, 1, 4))
    with pytest.raises(ValueError, match="matching shapes"):
        pack_kda_conv_weights(weights, dtype=torch.bfloat16, device="cpu")
