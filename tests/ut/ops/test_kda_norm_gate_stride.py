# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata/dispatch CPU tests for the existing common Kimi norm-gate kernel."""

from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ops.triton.kda import kda


@pytest.mark.parametrize("tokens", [1, 2, 4])
@pytest.mark.parametrize("row_width", [512, 1024, 1028, 2576])
@pytest.mark.parametrize("gate_rank", [3, 4])
@pytest.mark.parametrize("inplace", [False, True])
def test_packed_gate_uses_actual_token_stride_without_copy(tokens, row_width, gate_rank, inplace):
    heads, dim, offset = 4, 128, 16
    backing = torch.empty(offset + tokens * row_width + 31, dtype=torch.bfloat16)
    gate = backing.as_strided((tokens, heads, dim), (row_width, dim, 1), offset)
    if gate_rank == 4:
        gate = gate.unsqueeze(0)
    x = torch.empty(1, tokens, heads, dim, dtype=gate.dtype)
    out = x if inplace else torch.empty_like(x)
    weight = torch.empty(dim, dtype=x.dtype)
    with (
        patch.object(kda, "layer_norm_gated_fwd_kernel") as kernel,
        patch.object(kda, "layer_norm_gated_fwd", side_effect=AssertionError("compatible gate must stay fused")),
        patch.object(torch.Tensor, "reshape", side_effect=AssertionError("gate must not be materialized")),
        patch.object(torch.Tensor, "copy_", side_effect=AssertionError("direct output must not need copy")),
        patch.object(torch.Tensor, "item", side_effect=AssertionError("metadata only, no D2H")),
        patch.object(torch.UntypedStorage, "data_ptr", side_effect=AssertionError("no storage-pointer graph guards")),
    ):
        actual = kda.rms_norm_gated(x, gate, weight, None, "sigmoid", out=out)
    assert actual is out
    launch = kernel.__getitem__.return_value
    launch.assert_called_once()
    kwargs = launch.call_args.kwargs
    assert kwargs["x"] is x and kwargs["g"] is gate and kwargs["y"] is out
    assert kwargs["H"] == heads and kwargs["D"] == dim
    assert kwargs["T"] == tokens * heads
    assert kwargs["PACKED_GATE"] == (not gate.is_contiguous())
    if gate.is_contiguous() or row_width == kda._packed_bfg_row_width(heads, dim):
        assert "G_TOKEN_STRIDE" not in kwargs
    else:
        assert kwargs["G_TOKEN_STRIDE"] == row_width
    assert kwargs["STORE_RSTD"] is False and kwargs["STRIDED_GATE"] is False


@pytest.mark.parametrize(
    "layout", ["overlap_tokens", "strided_heads", "strided_features", "batch_gt_one", "wide_features"]
)
def test_unsupported_gates_retain_existing_fallback(layout):
    tokens, heads, dim = 2, 4, 128
    if layout == "wide_features":
        dim = 1024
    x = torch.empty(1, tokens, heads, dim, dtype=torch.bfloat16)
    backing = torch.empty(100000, dtype=x.dtype)
    strides = (heads * dim + 4, dim, 1)
    if layout == "overlap_tokens":
        strides = (heads * dim - 1, dim, 1)
    elif layout == "strided_heads":
        strides = (heads * (dim + 2) + 4, dim + 2, 1)
    elif layout == "strided_features":
        strides = (2 * heads * dim + 4, 2 * dim, 2)
    if layout == "batch_gt_one":
        x = torch.empty(2, tokens, heads, dim, dtype=x.dtype)
        gate = torch.empty_like(x)
    else:
        gate = backing.as_strided((tokens, heads, dim), strides, 16)
    y = torch.empty(x.numel() // dim, dim, dtype=x.dtype)
    with (
        patch.object(kda, "layer_norm_gated_fwd_kernel") as direct,
        patch.object(kda, "layer_norm_gated_fwd", return_value=(y, None, None, None)) as fallback,
    ):
        actual = kda.rms_norm_gated(x, gate, None, None, "sigmoid", out=x)
    assert actual is x
    direct.__getitem__.assert_not_called()
    fallback.assert_called_once()


@pytest.mark.parametrize("invalid", ["shape", "out_shape", "out_dtype", "out_stride", "weight", "bias"])
def test_generic_packed_gate_retains_direct_output_validation(invalid):
    x = torch.empty(1, 2, 4, 128, dtype=torch.bfloat16)
    gate = torch.empty(2, 2576, dtype=x.dtype)[:, 1024:1536].view(2, 4, 128)
    out = torch.empty_like(x)
    weight = torch.empty(128, dtype=x.dtype)
    bias = None
    if invalid == "shape":
        gate = gate[:1]
    elif invalid == "out_shape":
        out = out[:, :1]
    elif invalid == "out_dtype":
        out = out.float()
    elif invalid == "out_stride":
        out = torch.empty(1, 2, 4, 256, dtype=x.dtype)[..., ::2]
    elif invalid == "weight":
        weight = weight[:-1]
    else:
        bias = torch.empty(127, dtype=x.dtype)
    with patch.object(kda, "layer_norm_gated_fwd_kernel") as kernel, pytest.raises(ValueError):
        kda.rms_norm_gated(x, gate, weight, bias, "sigmoid", out=out)
    kernel.__getitem__.assert_not_called()


def test_2d_layer_norm_call_remains_without_token_stride_option():
    x = torch.empty(8, 128, dtype=torch.bfloat16)
    g = torch.empty(8, 256, dtype=x.dtype)[:, ::2]
    with patch.object(kda, "layer_norm_gated_fwd_kernel") as kernel:
        kda.layer_norm_gated_fwd(x, g, None, None, "sigmoid", is_rms_norm=True, return_stats=False)
    options = kernel.__getitem__.return_value.call_args.kwargs
    assert options["PACKED_GATE"] is False
    assert "G_TOKEN_STRIDE" not in options
    assert options["G_ROW_STRIDE"] == g.stride(0) and options["G_COL_STRIDE"] == g.stride(1)
