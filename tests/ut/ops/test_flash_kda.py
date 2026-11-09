# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for backend selection and the optional FlashKDA boundary."""

import builtins
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.ops import flash_kda, kda

_INTERNAL_FLASH_KDA_MODULE = "vllm_ascend.ops.pypto.flash_kda"


def _block_optional_imports(name, globals=None, locals=None, fromlist=(), level=0):
    if (
        name == "flash_kda"
        or name.startswith(("flash_kda.", "vllm_ascend.ops.pypto"))
        or name.split(".", 1)[0] in ("pypto", "pypto_pro")
    ):
        raise AssertionError(f"optional operator imported eagerly: {name}")
    return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)


_ORIGINAL_IMPORT = builtins.__import__


class _NPUTensor(torch.Tensor):
    """Keep real CPU arithmetic while representing an NPU boundary tensor."""

    @property
    def device(self):
        return torch.device("npu:0")


def _tensor(shape, dtype=torch.bfloat16, *, noncontiguous=False):
    if noncontiguous:
        value = torch.randn(*shape, 2, dtype=dtype)[..., 0]
    else:
        value = torch.randn(*shape, dtype=dtype)
    return value.as_subclass(_NPUTensor)


def _inputs(*, lengths=(63, 65), key_heads=4, value_heads=4, cache_dtype=torch.float32):
    tokens = sum(lengths)
    cu_seqlens = [0]
    for length in lengths:
        cu_seqlens.append(cu_seqlens[-1] + length)
    return dict(
        q=_tensor((1, tokens, key_heads, 128), noncontiguous=True),
        k=_tensor((1, tokens, key_heads, 128), noncontiguous=True),
        v=_tensor((1, tokens, value_heads, 128), noncontiguous=True),
        raw_gate=_tensor((1, tokens, value_heads, 128), noncontiguous=True),
        beta=_tensor((1, tokens, value_heads), noncontiguous=True),
        initial_state=_tensor((len(lengths), value_heads, 128, 128), cache_dtype, noncontiguous=True),
        cu_seqlens=tuple(cu_seqlens),
        chunk_indices=(),
        a_log=_tensor((1, value_heads), torch.float32, noncontiguous=True),
        dt_bias=_tensor((value_heads * 128,), torch.float32, noncontiguous=True),
        lower_bound=-5.0,
    )


@pytest.fixture
def a5_profile():
    with patch.object(
        flash_kda, "get_current_hardware_profile", return_value=get_hardware_profile(AscendDeviceType.A5)
    ):
        yield


def _load_backend():
    path = Path(flash_kda.__file__).with_name("kda_backend.py")
    spec = importlib.util.spec_from_file_location("_flash_kda_backend_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("device_type", list(AscendDeviceType))
@pytest.mark.parametrize("enabled", [False, True])
def test_import_selection_is_a5_only_and_does_not_import_vendor(monkeypatch, device_type, enabled):
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_FLASH_KDA", str(int(enabled)))
    with (
        patch("vllm_ascend.device.get_current_hardware_profile", return_value=get_hardware_profile(device_type)),
        patch("builtins.__import__", side_effect=_block_optional_imports),
    ):
        if enabled and device_type != AscendDeviceType.A5:
            with pytest.raises(RuntimeError, match="requires A5"):
                _load_backend()
        else:
            backend = _load_backend()
            assert backend.FLASH_KDA_ENABLED is enabled
            expected = flash_kda.run_flash_chunk_kda if enabled else kda.run_chunk_kda
            assert backend.run_chunk_kda is expected
            # Changing the environment after import must not change beta's ABI.
            monkeypatch.setenv("VLLM_ASCEND_ENABLE_FLASH_KDA", str(int(not enabled)))
            assert backend.FLASH_KDA_ENABLED is enabled
            assert backend.run_chunk_kda is expected


@pytest.mark.parametrize("lengths", [(100,), (63, 65), (128,)])
@pytest.mark.parametrize("key_heads,value_heads", [(4, 4), (16, 16), (4, 8)])
@pytest.mark.parametrize("cache_dtype", [torch.float32, torch.bfloat16])
def test_adapter_preserves_raw_inputs_and_vk_state(a5_profile, lengths, key_heads, value_heads, cache_dtype):
    inputs = _inputs(lengths=lengths, key_heads=key_heads, value_heads=value_heads, cache_dtype=cache_dtype)
    vendor = ModuleType(_INTERNAL_FLASH_KDA_MODULE)
    expected_output, expected_state = object(), object()
    vendor.flash_kda_wrapper = MagicMock(return_value=(expected_output, expected_state))
    tensor_constructor = torch.tensor

    def make_cu(data, *, dtype, device):
        assert device == torch.device("npu:0")
        return tensor_constructor(data, dtype=dtype).as_subclass(_NPUTensor)

    with (
        patch.dict(sys.modules, {_INTERNAL_FLASH_KDA_MODULE: vendor}),
        patch.object(flash_kda.torch, "tensor", side_effect=make_cu),
    ):
        assert flash_kda.run_flash_chunk_kda(**inputs) == (expected_output, expected_state)

    args, kwargs = vendor.flash_kda_wrapper.call_args
    for actual, name in zip(args, ("q", "k", "v", "raw_gate", "beta")):
        assert actual.is_contiguous()
        assert actual.dtype == torch.bfloat16
        torch.testing.assert_close(actual, inputs[name])
    assert kwargs["scale"] == 128**-0.5
    assert kwargs["layout_qkv"] == "BSND"
    assert kwargs["lower_bound"] == -5.0
    assert kwargs["initial_state"].shape == (len(lengths), value_heads, 128, 128)
    assert kwargs["a_log"].shape == (value_heads,)
    assert kwargs["dt_bias"].shape == (value_heads, 128)
    for name in ("initial_state", "a_log", "dt_bias"):
        assert kwargs[name].is_contiguous()
        assert kwargs[name].dtype == torch.float32
        torch.testing.assert_close(kwargs[name].reshape(-1), inputs[name].float().reshape(-1))
    assert kwargs["cu_seqlens"].dtype == torch.int64
    assert kwargs["cu_seqlens"].tolist() == list(inputs["cu_seqlens"])
    assert "chunk_indices" not in kwargs
    assert "use_beta_sigmoid_in_kernel" not in kwargs


@pytest.mark.parametrize("lower_bound", [None, -5.1, 0.1, float("nan")])
def test_adapter_rejects_unsupported_gate_without_importing_vendor(a5_profile, lower_bound):
    inputs = _inputs(lengths=(5,))
    inputs["lower_bound"] = lower_bound
    with (
        patch("builtins.__import__", side_effect=_block_optional_imports),
        pytest.raises(ValueError, match="safe_gate"),
    ):
        flash_kda.run_flash_chunk_kda(**inputs)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("q", lambda: _tensor((1, 5, 4, 64)), "D=128"),
        ("q", lambda: _tensor((2, 5, 4, 128)), "B=1"),
        ("beta", lambda: _tensor((1, 5, 4), torch.float32), "BF16"),
        ("raw_gate", lambda: _tensor((1, 5, 8, 128)), "gate"),
        ("initial_state", lambda: _tensor((2, 4, 128, 128)), "initial_state"),
        ("initial_state", lambda: _tensor((1, 4, 128, 128), torch.float16), "FP32 or BF16"),
        ("dt_bias", lambda: _tensor((127,)), "dt_bias"),
        ("a_log", lambda: _tensor((3,)), "A_log"),
        ("cu_seqlens", lambda: (0, 3), "span all packed tokens"),
        ("cu_seqlens", lambda: (0, 0, 5), "no empty sequences"),
        ("cu_seqlens", lambda: (0, 4.5, 5), "span all packed tokens"),
        ("cu_seqlens", lambda: torch.tensor([0, 5]), "host cu_seqlens"),
        ("q", lambda: torch.randn(1, 5, 4, 128, dtype=torch.bfloat16), "same NPU"),
    ],
)
def test_adapter_rejects_unsupported_contract(a5_profile, field, value, message):
    inputs = _inputs(lengths=(5,))
    inputs[field] = value()
    with pytest.raises(ValueError, match=message):
        flash_kda.run_flash_chunk_kda(**inputs)


def test_adapter_rejects_invalid_gqa(a5_profile):
    with pytest.raises(ValueError, match="Nv % Nk"):
        flash_kda.run_flash_chunk_kda(**_inputs(lengths=(5,), key_heads=3, value_heads=4))


@pytest.mark.parametrize("key_heads,value_heads", [(65, 65), (1, 97)])
def test_adapter_rejects_vendor_head_limits(a5_profile, key_heads, value_heads):
    with pytest.raises(ValueError, match="Nk=64 and Nv=96"):
        flash_kda.run_flash_chunk_kda(**_inputs(lengths=(1,), key_heads=key_heads, value_heads=value_heads))


def test_adapter_rejects_vendor_sequence_limit(a5_profile):
    with pytest.raises(ValueError, match="64"):
        flash_kda.run_flash_chunk_kda(**_inputs(lengths=(1,) * 65, key_heads=1, value_heads=1))


@pytest.mark.parametrize("device_type", [AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType._310P])
def test_direct_adapter_call_cannot_bypass_hardware_guard(device_type):
    with (
        patch.object(flash_kda, "get_current_hardware_profile", return_value=get_hardware_profile(device_type)),
        pytest.raises(RuntimeError, match="only on A5"),
    ):
        flash_kda.run_flash_chunk_kda(**_inputs(lengths=(5,)))


@pytest.mark.parametrize("missing", ["pypto", "pypto_pro", "pypto_pro.language"])
def test_missing_pypto_has_actionable_error(a5_profile, missing):
    inputs = _inputs(lengths=(5,))
    failure = ModuleNotFoundError(f"No module named '{missing}'", name=missing)

    def import_without_pypto(name, *args, **kwargs):
        if name == _INTERNAL_FLASH_KDA_MODULE:
            raise failure
        return _ORIGINAL_IMPORT(name, *args, **kwargs)

    with (
        patch("builtins.__import__", side_effect=import_without_pypto),
        pytest.raises(RuntimeError, match="PyPTO") as error,
    ):
        flash_kda.run_flash_chunk_kda(**inputs)
    assert error.value.__cause__ is failure
    assert "vendor flash_kda/" not in str(error.value)


@pytest.mark.parametrize("missing", [_INTERNAL_FLASH_KDA_MODULE, "torch_npu", "unrelated_dependency"])
def test_missing_non_pypto_dependency_is_not_mislabeled_or_fallback(a5_profile, missing):
    inputs = _inputs(lengths=(5,))
    failure = ModuleNotFoundError(f"No module named '{missing}'", name=missing)

    def import_with_unrelated_failure(name, *args, **kwargs):
        if name == _INTERNAL_FLASH_KDA_MODULE:
            raise failure
        return _ORIGINAL_IMPORT(name, *args, **kwargs)

    with (
        patch("builtins.__import__", side_effect=import_with_unrelated_failure),
        pytest.raises(ModuleNotFoundError) as error,
    ):
        flash_kda.run_flash_chunk_kda(**inputs)
    assert error.value is failure


def test_vendor_runtime_failure_is_not_silently_fallback(a5_profile):
    vendor = ModuleType(_INTERNAL_FLASH_KDA_MODULE)
    failure = RuntimeError("vendor kernel failed")
    vendor.flash_kda_wrapper = MagicMock(side_effect=failure)
    tensor_constructor = torch.tensor

    def make_cu(data, *, dtype, device):
        return tensor_constructor(data, dtype=dtype)

    with (
        patch.dict(sys.modules, {_INTERNAL_FLASH_KDA_MODULE: vendor}),
        patch.object(flash_kda.torch, "tensor", side_effect=make_cu),
        pytest.raises(RuntimeError, match="vendor kernel failed") as error,
    ):
        flash_kda.run_flash_chunk_kda(**_inputs(lengths=(5,)))
    assert error.value is failure


def test_prepared_cu_is_reused_without_allocation_or_device_read(a5_profile):
    inputs = _inputs(lengths=(63, 65))
    prepared = torch.tensor(inputs["cu_seqlens"], dtype=torch.int64).as_subclass(_NPUTensor)
    inputs["cu_seqlens_device"] = prepared
    vendor = ModuleType(_INTERNAL_FLASH_KDA_MODULE)
    vendor.flash_kda_wrapper = MagicMock(return_value=(object(), object()))
    with (
        patch.dict(sys.modules, {_INTERNAL_FLASH_KDA_MODULE: vendor}),
        patch.object(flash_kda.torch, "tensor", side_effect=AssertionError("per-layer cu allocation")),
        patch.object(_NPUTensor, "cpu", side_effect=AssertionError("device read")),
        patch.object(_NPUTensor, "tolist", side_effect=AssertionError("device read")),
        patch.object(_NPUTensor, "item", side_effect=AssertionError("device read")),
    ):
        for _ in range(3):
            flash_kda.run_flash_chunk_kda(**inputs)
            assert vendor.flash_kda_wrapper.call_args.kwargs["cu_seqlens"] is prepared
    assert vendor.flash_kda_wrapper.call_count == 3


def test_adapter_uses_internal_operator_without_external_flash_kda_package(a5_profile):
    inputs = _inputs(lengths=(5,))
    inputs["cu_seqlens_device"] = torch.tensor([0, 5], dtype=torch.int64).as_subclass(_NPUTensor)
    bundled = ModuleType(_INTERNAL_FLASH_KDA_MODULE)
    expected = (object(), object())
    bundled.flash_kda_wrapper = MagicMock(return_value=expected)

    def reject_external_operator(name, *args, **kwargs):
        if name == "flash_kda" or name.startswith("flash_kda."):
            raise AssertionError(f"external operator package required: {name}")
        return _ORIGINAL_IMPORT(name, *args, **kwargs)

    with (
        patch.dict(sys.modules, {_INTERNAL_FLASH_KDA_MODULE: bundled}),
        patch("builtins.__import__", side_effect=reject_external_operator),
        patch.object(flash_kda.torch, "tensor", side_effect=AssertionError("per-layer cu allocation")),
    ):
        assert flash_kda.run_flash_chunk_kda(**inputs) == expected
    bundled.flash_kda_wrapper.assert_called_once()


@pytest.mark.parametrize(
    "prepared",
    [
        lambda: torch.tensor([0, 5], dtype=torch.int64),
        lambda: torch.tensor([0, 5], dtype=torch.int32).as_subclass(_NPUTensor),
        lambda: torch.tensor([[0, 5]], dtype=torch.int64).as_subclass(_NPUTensor),
        lambda: torch.tensor([0, 3, 5], dtype=torch.int64).as_subclass(_NPUTensor),
        lambda: torch.tensor([0, 0, 5, 0], dtype=torch.int64)[::2].as_subclass(_NPUTensor),
        lambda: (0, 5),
    ],
)
def test_prepared_cu_rejects_incompatible_device_metadata(a5_profile, prepared):
    inputs = _inputs(lengths=(5,))
    inputs["cu_seqlens_device"] = prepared()
    with pytest.raises(ValueError, match="prepared cu_seqlens"):
        flash_kda.run_flash_chunk_kda(**inputs)


def test_prepared_cu_keeps_host_boundary_validation(a5_profile):
    inputs = _inputs(lengths=(5,))
    inputs["cu_seqlens_device"] = torch.tensor([0, 5], dtype=torch.int64).as_subclass(_NPUTensor)
    inputs["cu_seqlens"] = (0, 0, 5)
    with pytest.raises(ValueError, match="no empty sequences"):
        flash_kda.run_flash_chunk_kda(**inputs)
