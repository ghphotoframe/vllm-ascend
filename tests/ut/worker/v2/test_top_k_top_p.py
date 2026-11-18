# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import importlib
import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu.sample import sampler as upstream_sampler

from vllm_ascend.sample.sampler import _apply_top_k_top_p_pytorch

sampling = importlib.import_module("vllm_ascend.worker.v2.sample.apply_top_k_top_p")


def _assert_same_logits(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    # This also checks that unmasked logits are neither rounded nor reordered.
    assert torch.equal(actual, expected)


def _no_fallback(*args, **kwargs):
    raise AssertionError("small-k top-k-only filtering unexpectedly used full sorting")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch", [1, 2, 4])
def test_small_k_matches_legacy_without_full_sort(monkeypatch, dtype, batch):
    generator = torch.Generator().manual_seed(20261111)
    logits = torch.randn(batch, 257, generator=generator).to(dtype)
    values = [20] if batch == 1 else [1, 20, 64, 257][:batch]
    k = torch.tensor(values, dtype=torch.int32)
    original_k = k.clone()
    max_top_k = max(value for value in values if value != logits.shape[1])
    expected = _apply_top_k_top_p_pytorch(logits.clone(), k, None)
    rng_before = torch.random.get_rng_state()

    with monkeypatch.context() as patch:
        patch.setattr(sampling, "_apply_top_k_top_p_pytorch", _no_fallback)
        patch.setattr(torch.Tensor, "sort", _no_fallback)
        actual = sampling.apply_top_k_top_p_npu(logits.clone(), k, None, max_top_k=max_top_k)

    _assert_same_logits(actual, expected)
    assert torch.equal(k, original_k)
    assert torch.equal(torch.random.get_rng_state(), rng_before)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_probability_ties_and_underflow_keep_legacy_support(monkeypatch, dtype):
    logits = torch.tensor(
        [[3.0, 2.0, 2.0, 2.0, 0.0], [0.0, -1000.0, -1001.0, -1002.0, -1003.0], [1.0] * 5],
        dtype=dtype,
    )
    k = torch.tensor([2, 2, 1], dtype=torch.int32)
    expected = _apply_top_k_top_p_pytorch(logits.clone(), k, None)
    with monkeypatch.context() as patch:
        patch.setattr(sampling, "_apply_top_k_top_p_pytorch", _no_fallback)
        actual = sampling.apply_top_k_top_p_npu(logits.clone(), k, None, max_top_k=2)

    _assert_same_logits(actual, expected)
    assert torch.isfinite(actual).sum(dim=-1).tolist() == [4, 5, 5]
    # In the underflow row, the kth probability is zero. A logits-threshold
    # rewrite would incorrectly discard the last three finite logits.
    assert torch.equal(actual[1], logits[1])


def test_noncontiguous_logits_and_k_are_not_rewritten(monkeypatch):
    generator = torch.Generator().manual_seed(77)
    storage = torch.randn(4, 514, generator=generator)
    untouched_columns = storage[:, 1::2].clone()
    logits = storage[:, ::2]
    k_storage = torch.tensor([1, -10, 20, -10, 64, -10, 257, -10], dtype=torch.int32)
    original_k = k_storage.clone()
    k = k_storage[::2]
    assert not logits.is_contiguous() and not k.is_contiguous()
    expected = _apply_top_k_top_p_pytorch(logits.clone(), k.clone(), None)

    with monkeypatch.context() as patch:
        patch.setattr(sampling, "_apply_top_k_top_p_pytorch", _no_fallback)
        actual = sampling.apply_top_k_top_p_npu(logits, k, None, max_top_k=64)

    _assert_same_logits(actual, expected)
    assert torch.equal(storage[:, 1::2], untouched_columns)
    assert torch.equal(k_storage, original_k)


@pytest.mark.parametrize(
    ("k_values", "p_values", "hint"),
    [
        ([20, 20], None, None),
        ([65, 65], None, 65),
        ([20, 20], [1.0, 1.0], 20),
        ([20, 257], [0.9, 1.0], 20),
        (None, [0.7, 0.95], None),
    ],
)
def test_ineligible_cases_preserve_legacy_filtering(monkeypatch, k_values, p_values, hint):
    logits = torch.randn(2, 257, generator=torch.Generator().manual_seed(9))
    k = torch.tensor(k_values, dtype=torch.int32) if k_values is not None else None
    p = torch.tensor(p_values, dtype=torch.float32) if p_values is not None else None
    original_k = k.clone() if k is not None else None
    original_p = p.clone() if p is not None else None
    expected = _apply_top_k_top_p_pytorch(logits.clone(), k, p)
    fallback = Mock(wraps=_apply_top_k_top_p_pytorch)
    monkeypatch.setattr(sampling, "_apply_top_k_top_p_pytorch", fallback)

    actual = sampling.apply_top_k_top_p_npu(logits.clone(), k, p, max_top_k=hint)

    fallback.assert_called_once()
    _assert_same_logits(actual, expected)
    if k is not None:
        assert torch.equal(k, original_k)
    if p is not None:
        assert torch.equal(p, original_p)


def test_no_filter_returns_the_original_tensor(monkeypatch):
    logits = torch.randn(2, 257)
    monkeypatch.setattr(sampling, "_apply_top_k_top_p_pytorch", _no_fallback)
    assert sampling.apply_top_k_top_p_npu(logits, None, None) is logits


def _states(k_cpu, k_device, p_device=None):
    return SimpleNamespace(
        vocab_size=257,
        top_k=SimpleNamespace(np=np.asarray(k_cpu, dtype=np.int32)),
        get_top_k_top_p=Mock(return_value=(k_device, p_device)),
    )


@pytest.mark.parametrize("active_values", [(20, 1), (257, 20)])
def test_cpu_hint_ignores_inactive_slots_and_no_top_k_rows(monkeypatch, active_values):
    k_cpu = [257, active_values[1], 200, 128, active_values[0], 257, 256]
    idx_mapping_np = np.array([4, 1], dtype=np.int32)
    # Expanded logits can repeat a request during speculative verification.
    expanded_idx_mapping = torch.tensor([4, 4, 1, 1, 4], dtype=torch.int32)
    k_device = torch.tensor(np.asarray(k_cpu)[[4, 4, 1, 1, 4]], dtype=torch.int32)
    state = _states(k_cpu, k_device)
    logits = torch.randn(5, 257)
    calls = []

    def filter_spy(values, k, p, *, max_top_k=None):
        calls.append((values, k, p, max_top_k))
        return values

    monkeypatch.setattr(sampling, "apply_top_k_top_p_npu", filter_spy)
    actual = sampling.apply_top_k_top_p_with_cpu_metadata(state, logits, expanded_idx_mapping, idx_mapping_np)

    assert actual is logits
    assert len(calls) == 1
    assert calls[0][0] is logits and calls[0][1] is k_device and calls[0][2] is None
    assert calls[0][3] == 20
    state.get_top_k_top_p.assert_called_once_with(expanded_idx_mapping, idx_mapping_np)


def test_cpu_metadata_and_fast_filter_do_not_read_device_scalars(monkeypatch):
    k_cpu = [257, 20, 257, 1, 257]
    expanded = torch.tensor([1, 3, 1], dtype=torch.int32)
    idx_mapping_np = np.array([3, 1], dtype=np.int32)
    k_device = torch.tensor([20, 1, 20], dtype=torch.int32)
    state = _states(k_cpu, k_device)
    logits = torch.randn(3, 257, generator=torch.Generator().manual_seed(8))
    expected = _apply_top_k_top_p_pytorch(logits.clone(), k_device, None)

    def forbidden(*args, **kwargs):
        raise AssertionError("sampling attempted a tensor-to-host scalar read")

    with monkeypatch.context() as patch:
        patch.setattr(sampling, "_apply_top_k_top_p_pytorch", _no_fallback)
        for name in ("item", "cpu", "numpy", "tolist", "__int__", "__index__", "__bool__"):
            patch.setattr(torch.Tensor, name, forbidden)
        actual = sampling.apply_top_k_top_p_with_cpu_metadata(state, logits.clone(), expanded, idx_mapping_np)

    _assert_same_logits(actual, expected)


@pytest.mark.parametrize("empty", [False, True])
def test_cpu_metadata_no_filter_skips_reduction_and_filter(monkeypatch, empty):
    logits = torch.empty(0 if empty else 2, 257)
    indices = np.array([], dtype=np.int32) if empty else np.array([0, 2], dtype=np.int32)
    expanded = torch.tensor([], dtype=torch.int32) if empty else torch.tensor([0, 2], dtype=torch.int32)
    state = _states([257, 257, 257], None)
    monkeypatch.setattr(sampling, "apply_top_k_top_p_npu", _no_fallback)

    assert sampling.apply_top_k_top_p_with_cpu_metadata(state, logits, expanded, indices) is logits


def _original_sample_spy(calls):
    def original(self, processed_logits, expanded_idx_mapping, idx_mapping_np, pos, top_k, top_p, use_flashinfer):
        tokens = torch.zeros(processed_logits.shape[0], dtype=torch.int64)
        calls.append(
            SimpleNamespace(
                owner=self,
                logits=processed_logits,
                expanded=expanded_idx_mapping,
                indices=idx_mapping_np,
                pos=pos,
                k=top_k,
                p=top_p,
                flashinfer=use_flashinfer,
                sampled=tokens,
            )
        )
        return tokens, processed_logits

    return original


def test_random_sampler_fast_path_preserves_original_categorical_contract(monkeypatch):
    logits = torch.randn(5, 257, generator=torch.Generator().manual_seed(41))
    expanded = torch.tensor([4, 4, 1, 1, 4], dtype=torch.int32)
    indices = np.array([4, 1], dtype=np.int32)
    pos = torch.tensor([11, 12, 30, 31, 13], dtype=torch.int32)
    k = torch.tensor([20, 20, 257, 257, 20], dtype=torch.int32)
    state = _states([128, 257, 200, 64, 20, 257], k)
    sampler = SimpleNamespace(sampling_states=state, use_fp64_gumbel=True)
    expected = _apply_top_k_top_p_pytorch(logits.clone(), k, None)
    calls = []
    sample_random = sampling.wrap_sample_random_with_cpu_metadata(_original_sample_spy(calls))
    monkeypatch.setattr(sampling, "_apply_top_k_top_p_pytorch", _no_fallback)
    rng_before = torch.random.get_rng_state()

    sampled, processed = sample_random(sampler, logits, expanded, indices, pos, k, None, False)

    assert len(calls) == 1
    call = calls[0]
    assert call.owner is sampler
    assert call.expanded is expanded and call.indices is indices and call.pos is pos
    # The existing full-vocabulary categorical implementation consumes RNG;
    # the replacement only pre-filters and prevents a second filtering pass.
    assert call.k is None and call.p is None and call.flashinfer is False
    assert call.logits.shape == (5, 257)
    assert sampled is call.sampled and processed is call.logits
    assert sampler.use_fp64_gumbel is True
    _assert_same_logits(processed, expected)
    assert torch.equal(torch.random.get_rng_state(), rng_before)


@pytest.mark.parametrize(
    ("k_values", "p_values", "use_flashinfer"),
    [
        ([20, 20], None, True),
        ([65, 257], None, False),
        ([20, 20], [0.9, 1.0], False),
        (None, [0.8, 0.9], False),
        (None, None, False),
    ],
)
def test_random_sampler_fallback_delegates_original_arguments(monkeypatch, k_values, p_values, use_flashinfer):
    logits = torch.randn(2, 257, generator=torch.Generator().manual_seed(101))
    original_logits = logits.clone()
    k = torch.tensor(k_values, dtype=torch.int32) if k_values is not None else None
    p = torch.tensor(p_values, dtype=torch.float32) if p_values is not None else None
    host_k = k_values if k_values is not None else [257, 257]
    sampler = SimpleNamespace(sampling_states=_states(host_k, k, p))
    expanded = torch.tensor([0, 1], dtype=torch.int32)
    indices = np.array([0, 1], dtype=np.int32)
    pos = torch.tensor([100, 200], dtype=torch.int32)
    calls = []
    sample_random = sampling.wrap_sample_random_with_cpu_metadata(_original_sample_spy(calls))
    monkeypatch.setattr(sampling, "apply_top_k_top_p_npu", _no_fallback)

    sampled, processed = sample_random(sampler, logits, expanded, indices, pos, k, p, use_flashinfer)

    assert len(calls) == 1
    call = calls[0]
    assert call.owner is sampler and call.logits is logits
    assert call.expanded is expanded and call.indices is indices and call.pos is pos
    assert call.k is k and call.p is p and call.flashinfer is use_flashinfer
    assert sampled is call.sampled and processed is logits
    assert torch.equal(logits, original_logits)


def test_random_sampler_fast_path_uses_cpu_hint_without_tensor_sync(monkeypatch):
    logits = torch.randn(2, 257, generator=torch.Generator().manual_seed(88))
    k = torch.tensor([20, 1], dtype=torch.int32)
    sampler = SimpleNamespace(sampling_states=_states([257, 1, 257, 20, 257], k))
    expanded = torch.tensor([3, 1], dtype=torch.int32)
    indices = np.array([3, 1], dtype=np.int32)
    pos = torch.tensor([11, 33], dtype=torch.int32)
    expected = _apply_top_k_top_p_pytorch(logits.clone(), k, None)
    # Allocate the spy result before forbidding host reads, so only the
    # production wrapper/filter are constrained by the synchronization guard.
    returned_tokens = torch.zeros(2, dtype=torch.int64)
    seen = []

    def original(self, values, expanded_arg, indices_arg, pos_arg, top_k, top_p, use_flashinfer):
        seen.append((self, expanded_arg, indices_arg, pos_arg, top_k, top_p, use_flashinfer))
        return returned_tokens, values

    def forbidden(*args, **kwargs):
        raise AssertionError("sampling attempted a tensor-to-host scalar read")

    sample_random = sampling.wrap_sample_random_with_cpu_metadata(original)
    with monkeypatch.context() as patch:
        patch.setattr(sampling, "_apply_top_k_top_p_pytorch", _no_fallback)
        for name in ("item", "cpu", "numpy", "tolist", "__int__", "__index__", "__bool__"):
            patch.setattr(torch.Tensor, name, forbidden)
        sampled, processed = sample_random(sampler, logits, expanded, indices, pos, k, None, False)

    assert sampled is returned_tokens
    assert len(seen) == 1 and seen[0][0] is sampler
    assert seen[0][1] is expanded and seen[0][2] is indices and seen[0][3] is pos
    assert seen[0][4:] == (None, None, False)
    _assert_same_logits(processed, expected)


def test_fast_path_delegates_to_real_upstream_categorical_entry(monkeypatch):
    logits = torch.randn(3, 257, generator=torch.Generator().manual_seed(29))
    expanded = torch.tensor([3, 1, 3], dtype=torch.int32)
    indices = np.array([3, 1], dtype=np.int32)
    pos = torch.tensor([21, 44, 22], dtype=torch.int32)
    top_k = torch.tensor([20, 1, 20], dtype=torch.int32)
    states = _states([257, 1, 257, 20, 257], top_k)
    temperatures = torch.tensor([1.0, 0.0, 1.0, 0.7, 1.0])
    seeds = torch.tensor([100, 200, 300, 400, 500], dtype=torch.int64)
    states.temperature = SimpleNamespace(gpu=temperatures)
    states.seeds = SimpleNamespace(gpu=seeds)
    owner = SimpleNamespace(sampling_states=states, use_fp64_gumbel=True)
    expected = _apply_top_k_top_p_pytorch(logits.clone(), top_k, None)

    # The class may already be patched by worker startup. Unwrap to exercise
    # upstream's real implementation rather than a stub of its RNG entry.
    original = inspect.unwrap(upstream_sampler.Sampler._sample_random)
    sample_random = sampling.wrap_sample_random_with_cpu_metadata(original)
    no_op_filter = Mock(wraps=upstream_sampler.apply_top_k_top_p)
    chosen = torch.tensor([11, 22, 33], dtype=torch.int64)
    categorical = Mock(return_value=chosen)
    monkeypatch.setattr(upstream_sampler, "apply_top_k_top_p", no_op_filter)
    monkeypatch.setattr(upstream_sampler, "gumbel_sample", categorical)
    monkeypatch.setattr(sampling, "_apply_top_k_top_p_pytorch", _no_fallback)

    sampled, processed = sample_random(owner, logits, expanded, indices, pos, top_k, None, False)

    assert sampled is chosen and processed is logits
    _assert_same_logits(processed, expected)
    no_op_filter.assert_called_once()
    passed_logits, passed_k, passed_p = no_op_filter.call_args.args
    assert passed_logits is processed and passed_k is None and passed_p is None
    categorical.assert_called_once()
    args, kwargs = categorical.call_args
    assert args[0] is processed and args[0].shape == (3, 257)
    assert args[1] is expanded
    assert args[2] is temperatures and args[3] is seeds and args[4] is pos
    assert kwargs == {"apply_temperature": False, "is_drafting": False, "use_fp64": True}
