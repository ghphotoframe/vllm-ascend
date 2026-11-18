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


def test_native_decode_preparation_public_wrapper_preserves_five_argument_default():
    indices = torch.arange(2, dtype=torch.int32)
    starts = torch.arange(3, dtype=torch.int32)
    output = torch.empty(2, 5)
    with patch("vllm_ascend.ops.triton.kda.conv_state.prepare_native_conv_decode") as inner:
        result = prepare_native_conv_decode(indices, starts, 8, 0, output)
    inner.assert_called_once_with(indices, starts, 8, 0, output)
    assert result is inner.return_value


@pytest.mark.parametrize("rows,gap", [(1, 4), (2, 4), (4, 1040)])
def test_native_decode_preparation_optional_pack_passes_same_width_integer_views(rows, gap):
    width = 1536
    indices = torch.arange(2 * rows + 1, dtype=torch.int32)[1::2]
    starts = torch.arange(2 * (rows + 1) + 1, dtype=torch.int32)[1::2]
    backing = torch.empty(16 + rows * (width + gap), dtype=torch.bfloat16)
    source = backing.as_strided((rows, width), (width + gap, 1), 16)
    packed = torch.empty_like(source, memory_format=torch.contiguous_format)
    output = torch.empty_like(packed)
    with (
        patch("vllm_ascend.ops.triton.kda.conv_state._prepare_native_conv_decode_kernel") as kernel,
        patch.object(torch.Tensor, "item", side_effect=AssertionError("preparation cannot read host data")),
    ):
        normalized = prepare_native_conv_decode(indices, starts, 8, 0, output, input_tensor=source, packed_input=packed)
    launch = kernel.__getitem__.return_value
    launch.assert_called_once()
    options = launch.call_args.kwargs
    assert options["PACK_INPUT"] is True
    assert options["INPUT_ROW_STRIDE"] == width + gap
    assert options["INPUT"].dtype == options["PACKED_INPUT"].dtype == torch.int16
    assert options["INPUT"].stride() == source.stride()
    assert options["INPUT"].storage_offset() == source.storage_offset()
    assert options["INPUT"].untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
    assert options["PACKED_INPUT"].untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()
    assert normalized.dtype == torch.int32 and normalized.shape == (rows,)


@pytest.mark.parametrize("alias", ["source_packed", "source_output", "packed_output"])
def test_native_decode_preparation_compaction_rejects_shared_storage_even_for_disjoint_views(alias):
    rows, width = 2, 12
    indices = torch.arange(rows, dtype=torch.int32)
    starts = torch.arange(rows + 1, dtype=torch.int32)
    source = torch.empty(rows, width, dtype=torch.bfloat16)
    packed = torch.empty_like(source)
    output = torch.empty_like(source)
    shared = torch.empty(2 * rows * width, dtype=torch.bfloat16)
    first, second = shared[: rows * width].view(rows, width), shared[rows * width :].view(rows, width)
    if alias == "source_packed":
        source, packed = first, second
    elif alias == "source_output":
        source, output = first, second
    else:
        packed, output = first, second
    with (
        patch("vllm_ascend.ops.triton.kda.conv_state._prepare_native_conv_decode_kernel") as kernel,
        pytest.raises(ValueError, match="separate storage"),
    ):
        prepare_native_conv_decode(indices, starts, 8, 0, output, input_tensor=source, packed_input=packed)
    kernel.__getitem__.assert_not_called()


@pytest.mark.parametrize(
    "invalid", ["source_only", "destination_only", "source_dtype", "shape", "column_stride", "destination_stride"]
)
def test_native_decode_preparation_compaction_rejects_invalid_contracts(invalid):
    indices, starts = torch.arange(2, dtype=torch.int32), torch.arange(3, dtype=torch.int32)
    output = torch.empty(2, 12, dtype=torch.bfloat16)
    source, packed = torch.empty_like(output), torch.empty_like(output)
    if invalid == "source_only":
        packed = None
    elif invalid == "destination_only":
        source = None
    elif invalid == "source_dtype":
        source = source.float()
    elif invalid == "shape":
        source = source[:, :-1]
    elif invalid == "column_stride":
        source = torch.empty(2, 24, dtype=torch.bfloat16)[:, ::2]
    else:
        packed = torch.empty(2, 24, dtype=torch.bfloat16)[:, :12]
    with (
        patch("vllm_ascend.ops.triton.kda.conv_state._prepare_native_conv_decode_kernel") as kernel,
        pytest.raises(ValueError),
    ):
        prepare_native_conv_decode(indices, starts, 8, 0, output, input_tensor=source, packed_input=packed)
    kernel.__getitem__.assert_not_called()


def test_native_decode_preparation_empty_compaction_never_launches_or_checks_storage():
    indices = torch.empty(0, dtype=torch.int32)
    starts = torch.zeros(1, dtype=torch.int32)
    source, packed, output = (torch.empty(0, 12, dtype=torch.bfloat16) for _ in range(3))
    with (
        patch("vllm_ascend.ops.triton.kda.conv_state._prepare_native_conv_decode_kernel") as kernel,
        patch.object(torch.UntypedStorage, "data_ptr", side_effect=AssertionError("empty requests do not race")),
    ):
        result = prepare_native_conv_decode(indices, starts, 0, 0, output, input_tensor=source, packed_input=packed)
    assert result.shape == (0,)
    kernel.__getitem__.assert_not_called()


def _raw_gate_prepare_inputs(rows=2, width=8):
    stride, offset = 5 * width + 16, 16
    backing = torch.arange(offset + rows * stride + 64, dtype=torch.float32).bfloat16()
    qkv = backing.as_strided((rows, 3 * width), (stride, 1), offset)
    gate = backing.as_strided((rows, width), (stride, 1), offset + 3 * width)
    return dict(
        cache_indices=torch.arange(1, rows + 1, dtype=torch.int32),
        query_start_loc=torch.arange(rows + 1, dtype=torch.int32),
        num_slots=8,
        null_block_id=0,
        output=torch.empty(rows, 3 * width, dtype=torch.bfloat16),
        input_tensor=qkv,
        packed_input=torch.empty(rows, 3 * width, dtype=torch.bfloat16),
        gate_tensor=gate,
        packed_gate=torch.empty(rows, width, dtype=torch.bfloat16),
    )


@pytest.mark.parametrize("rows", [1, 2, 4])
@pytest.mark.parametrize("strided_metadata", [False, True])
def test_gate_prepare_uses_integer_views_and_allows_shared_read_only_sources(rows, strided_metadata):
    values = _raw_gate_prepare_inputs(rows)
    if strided_metadata:
        for key in ("cache_indices", "query_start_loc"):
            value = values[key]
            backing = torch.empty(2 * value.numel() + 1, dtype=value.dtype)
            backing[1::2].copy_(value)
            values[key] = backing[1::2]
    with (
        patch("vllm_ascend.ops.triton.kda.conv_state._prepare_native_conv_decode_kernel") as kernel,
        patch.object(torch.Tensor, "item", side_effect=AssertionError("no metadata D2H")),
    ):
        normalized = prepare_native_conv_decode(**values)
    options = kernel.__getitem__.return_value.call_args.kwargs
    assert options["PACK_INPUT"] is options["PACK_GATE"] is True
    assert options["GATE_INPUT"].dtype == options["PACKED_GATE"].dtype == torch.int16
    assert options["GATE_INPUT"].shape == values["gate_tensor"].shape
    assert options["GATE_INPUT"].storage_offset() == values["gate_tensor"].storage_offset()
    assert options["GATE_INPUT"].untyped_storage().data_ptr() == values["input_tensor"].untyped_storage().data_ptr()
    assert options["GATE_ROW_STRIDE"] == values["gate_tensor"].stride(0)
    assert options["PACKED_GATE"].untyped_storage().data_ptr() == values["packed_gate"].untyped_storage().data_ptr()
    assert normalized.dtype == torch.int32 and normalized.shape == (rows,)


@pytest.mark.parametrize("which", ["none", "qkv"])
def test_gate_prepare_default_does_not_add_gate_options_to_existing_kernel_call(which):
    values = _raw_gate_prepare_inputs()
    values.pop("gate_tensor")
    values.pop("packed_gate")
    if which == "none":
        values.pop("input_tensor")
        values.pop("packed_input")
    with patch("vllm_ascend.ops.triton.kda.conv_state._prepare_native_conv_decode_kernel") as kernel:
        prepare_native_conv_decode(**values)
    options = kernel.__getitem__.return_value.call_args.kwargs
    assert "GATE_INPUT" not in options and "PACK_GATE" not in options


@pytest.mark.parametrize(
    "alias",
    [
        "out_gate_source",
        "qkv_target_gate_source",
        "gate_target_gate_source",
        "gate_target_qkv_source",
        "gate_target_out",
        "gate_target_qkv_target",
        "gate_target_indices",
        "gate_target_starts",
    ],
)
def test_gate_prepare_rejects_write_aliases_but_not_the_shared_read_backing(alias):
    values = _raw_gate_prepare_inputs()
    shape = values["packed_gate"].shape
    if alias == "out_gate_source":
        values["output"] = values["gate_tensor"].as_strided(values["output"].shape, (24, 1), 128)
    elif alias == "qkv_target_gate_source":
        values["packed_input"] = values["gate_tensor"].as_strided(values["packed_input"].shape, (24, 1), 128)
    elif alias == "gate_target_gate_source":
        values["packed_gate"] = values["gate_tensor"].as_strided(shape, (8, 1), 128)
    elif alias == "gate_target_qkv_source":
        values["packed_gate"] = values["input_tensor"].as_strided(shape, (8, 1), 128)
    elif alias == "gate_target_out":
        values["packed_gate"] = values["output"].view(-1)[:16].view(shape)
    elif alias == "gate_target_qkv_target":
        values["packed_gate"] = values["packed_input"].view(-1)[:16].view(shape)
    else:
        backing = torch.empty(128, dtype=torch.int32)
        key = "cache_indices" if alias == "gate_target_indices" else "query_start_loc"
        target = values[key]
        values[key] = backing[: target.numel()]
        values[key].copy_(target)
        values["packed_gate"] = backing.view(torch.bfloat16)[:16].view(shape)
    with (
        patch("vllm_ascend.ops.triton.kda.conv_state._prepare_native_conv_decode_kernel") as kernel,
        pytest.raises(ValueError, match="separate storage"),
    ):
        prepare_native_conv_decode(**values)
    kernel.__getitem__.assert_not_called()


@pytest.mark.parametrize(
    "invalid",
    [
        "source_only",
        "target_only",
        "gate_only",
        "shape",
        "source_dtype",
        "target_dtype",
        "column_stride",
        "row_overlap",
        "target_stride",
        "qkv_width",
    ],
)
def test_gate_prepare_rejects_unsupported_contracts_before_launch(invalid):
    values = _raw_gate_prepare_inputs()
    if invalid == "source_only":
        values.pop("packed_gate")
    elif invalid == "target_only":
        values.pop("gate_tensor")
    elif invalid == "gate_only":
        values.pop("input_tensor")
        values.pop("packed_input")
    elif invalid == "shape":
        values["gate_tensor"] = values["gate_tensor"][:, :-1]
    elif invalid == "source_dtype":
        values["gate_tensor"] = values["gate_tensor"].float()
    elif invalid == "target_dtype":
        values["packed_gate"] = values["packed_gate"].float()
    elif invalid == "column_stride":
        values["gate_tensor"] = torch.empty(2, 16, dtype=torch.bfloat16)[:, ::2]
    elif invalid == "row_overlap":
        values["gate_tensor"] = torch.empty(2, 8, dtype=torch.bfloat16).as_strided((2, 8), (4, 1))
    elif invalid == "target_stride":
        values["packed_gate"] = torch.empty(2, 16, dtype=torch.bfloat16)[:, :8]
    else:
        values["output"] = values["output"][:, :-1]
        values["input_tensor"] = values["input_tensor"][:, :-1]
        values["packed_input"] = torch.empty(2, 23, dtype=torch.bfloat16)
    with (
        patch("vllm_ascend.ops.triton.kda.conv_state._prepare_native_conv_decode_kernel") as kernel,
        pytest.raises(ValueError),
    ):
        prepare_native_conv_decode(**values)
    kernel.__getitem__.assert_not_called()


def test_gate_prepare_empty_requests_keep_the_existing_no_launch_contract():
    values = _raw_gate_prepare_inputs(rows=0)
    with patch("vllm_ascend.ops.triton.kda.conv_state._prepare_native_conv_decode_kernel") as kernel:
        normalized = prepare_native_conv_decode(**values)
    assert normalized.numel() == 0
    kernel.__getitem__.assert_not_called()


def test_gate_preparation_public_wrapper_keeps_existing_return_and_forwards_only_explicit_options():
    values = _raw_gate_prepare_inputs()
    with patch("vllm_ascend.ops.triton.kda.conv_state.prepare_native_conv_decode") as inner:
        result = prepare_native_conv_decode(**values)
    inner.assert_called_once_with(
        values["cache_indices"],
        values["query_start_loc"],
        values["num_slots"],
        values["null_block_id"],
        values["output"],
        input_tensor=values["input_tensor"],
        packed_input=values["packed_input"],
        gate_tensor=values["gate_tensor"],
        packed_gate=values["packed_gate"],
    )
    assert result is inner.return_value
