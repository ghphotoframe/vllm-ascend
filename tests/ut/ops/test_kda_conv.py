# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ops.kda_conv import pack_kda_conv_weights, prepare_native_conv_decode


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


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("transpose_output", [False, True])
def test_native_decode_preparation_retains_view_offsets_and_strides(index_dtype, transpose_output):
    indices = torch.arange(8, dtype=index_dtype)[1:7:2]
    starts = torch.arange(10, dtype=torch.int64)[1:9:2]
    backing = torch.empty(8, 20)
    output = backing[1:4, 2:17:3] if not transpose_output else backing[1:6, 2:5].t()
    assert output.storage_offset() > 0

    with (
        patch("vllm_ascend.ops.triton.kda.conv_state._prepare_native_conv_decode_kernel") as kernel,
        patch.object(torch.Tensor, "item", side_effect=AssertionError("preparation cannot read tensor data on host")),
    ):
        normalized = prepare_native_conv_decode(indices, starts, 32, 0, output)

    launch = kernel.__getitem__.return_value
    launch.assert_called_once()
    args = launch.call_args.args
    assert args[:4] == (indices, starts, normalized, output)
    assert args[4:9] == (indices.stride(0), starts.stride(0), *output.stride(), 32)
    assert launch.call_args.kwargs["REQUESTS"] == indices.numel()
    assert launch.call_args.kwargs["DIM"] == output.shape[1]
    assert launch.call_args.kwargs["NULL_SLOT"] == 0
    assert normalized.shape == indices.shape
    assert normalized.dtype == index_dtype and normalized.device == indices.device
    assert normalized.is_contiguous()


def test_native_decode_preparation_empty_requests_do_not_launch():
    indices = torch.empty(0, dtype=torch.int32)
    starts = torch.zeros(1, dtype=torch.int32)
    with patch("vllm_ascend.ops.triton.kda.conv_state._prepare_native_conv_decode_kernel") as kernel:
        normalized = prepare_native_conv_decode(indices, starts, 0, 0, torch.empty(0, 5))

    kernel.__getitem__.assert_not_called()
    assert normalized.shape == (0,) and normalized.dtype == indices.dtype


@pytest.mark.parametrize(
    "invalid",
    ["index_rank", "index_dtype", "starts_shape", "starts_dtype", "output_shape", "output_overlap", "negative_slots"],
)
def test_native_decode_preparation_rejects_invalid_tensor_contracts(invalid):
    indices = torch.arange(3, dtype=torch.int32)
    starts = torch.arange(4, dtype=torch.int32)
    output = torch.empty(3, 5)
    num_slots = 8
    if invalid == "index_rank":
        indices = indices.unsqueeze(1)
    elif invalid == "index_dtype":
        indices = indices.float()
    elif invalid == "starts_shape":
        starts = starts[:-1]
    elif invalid == "starts_dtype":
        starts = starts.float()
    elif invalid == "output_shape":
        output = output[:-1]
    elif invalid == "output_overlap":
        output = output[:1].expand(3, 5)
    else:
        num_slots = -1

    with (
        patch("vllm_ascend.ops.triton.kda.conv_state._prepare_native_conv_decode_kernel") as kernel,
        pytest.raises(ValueError),
    ):
        prepare_native_conv_decode(indices, starts, num_slots, 0, output)
    kernel.__getitem__.assert_not_called()
