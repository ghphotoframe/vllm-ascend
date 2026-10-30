# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_ascend.ascend_forward_context import (
    MoECommType,
    get_mrv2_in_profile_run,
    get_mrv2_profile_has_prefill,
    override_mrv2_in_profile_run,
)
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.worker.v2.attn_utils import AscendAttentionMetadataMap, build_attn_metadata
from vllm_ascend.worker.v2.input_batch import AscendInputBatch, AscendInputBuffers
from vllm_ascend.worker.v2.model_runner import NPUModelRunner
from vllm_ascend.worker.v2.model_states.mamba_hybrid import AscendMambaHybridModelState


def _build_phase_metadata(has_prefill, *, for_capture=False):
    """Use one-token rows so query length cannot reveal request phase."""
    return build_attn_metadata(
        attn_groups=[],
        num_reqs=1,
        num_tokens=1,
        query_start_loc_gpu=torch.tensor([0, 1], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        max_query_len=1,
        seq_lens=torch.tensor([16], dtype=torch.int32),
        max_seq_len=16,
        block_tables=(),
        slot_mappings=torch.empty((0, 1), dtype=torch.int32),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        seq_lens_np=np.array([16], dtype=np.int32),
        has_prefill=has_prefill,
        for_cudagraph_capture=for_capture,
    )


@pytest.mark.parametrize("has_prefill", [True, False, None])
@pytest.mark.parametrize("for_capture", [True, False])
def test_phase_metadata_does_not_add_a_fake_attention_layer(has_prefill, for_capture):
    metadata = _build_phase_metadata(has_prefill, for_capture=for_capture)
    layer_metadata = object()
    metadata["layer.0"] = layer_metadata
    assert metadata.has_prefill is has_prefill
    assert list(metadata) == ["layer.0"]
    assert list(metadata.values()) == [layer_metadata]
    assert dict(metadata) == {"layer.0": layer_metadata}


@pytest.mark.parametrize("prefilling", [[True], [False], [False, True]])
def test_mamba_metadata_uses_request_phase_for_short_prefill_and_mixed_batch(prefilling):
    num_reqs = len(prefilling)
    batch = SimpleNamespace(
        num_reqs=num_reqs,
        num_reqs_after_padding=num_reqs,
        num_tokens=num_reqs,
        num_tokens_after_padding=num_reqs,
        is_prefilling_np=np.asarray(prefilling),
        has_prefill=any(prefilling),
        num_scheduled_tokens=np.ones(num_reqs, dtype=np.int32),
        query_start_loc=torch.arange(num_reqs + 1, dtype=torch.int32),
        query_start_loc_np=np.arange(num_reqs + 1, dtype=np.int32),
        seq_lens=torch.full((num_reqs,), 16, dtype=torch.int32),
        seq_lens_np=np.full(num_reqs, 16, dtype=np.int32),
        dcp_local_seq_lens=None,
        positions=torch.full((num_reqs,), 15, dtype=torch.int64),
        attn_state=None,
    )
    state = SimpleNamespace(
        vllm_config=SimpleNamespace(num_speculative_tokens=0, parallel_config=SimpleNamespace()),
        max_model_len=16,
        get_attn_rotary_positions=lambda _: None,
    )
    metadata = AscendMambaHybridModelState.prepare_attn(
        state,
        batch,
        CUDAGraphMode.NONE,
        (),
        torch.empty((0, num_reqs), dtype=torch.int32),
        [],
        SimpleNamespace(kv_cache_groups=[]),
    )
    assert metadata.has_prefill is any(prefilling)


def test_decode_capture_dummy_carries_false_phase():
    buffers = AscendInputBuffers(4, 4, torch.device("cpu"))
    with patch("vllm_ascend.worker.v2.input_batch.update_cos_sin"):
        batch = AscendInputBatch.make_dummy(4, 4, buffers)
    assert batch.has_prefill is False
    state = SimpleNamespace(
        vllm_config=SimpleNamespace(num_speculative_tokens=0, parallel_config=SimpleNamespace()),
        max_model_len=16,
        get_attn_rotary_positions=lambda _: None,
    )
    metadata = AscendMambaHybridModelState.prepare_attn(
        state,
        batch,
        CUDAGraphMode.NONE,
        (),
        torch.empty((0, 4), dtype=torch.int32),
        [],
        SimpleNamespace(kv_cache_groups=[]),
        for_capture=True,
    )
    assert metadata.has_prefill is False


@pytest.mark.parametrize(
    "metadata_phase,profile_phase,mode,expected_phase,comm",
    [
        (True, None, CUDAGraphMode.NONE, True, MoECommType.FUSED_MC2),
        (False, None, CUDAGraphMode.NONE, False, MoECommType.MC2),
        (False, None, CUDAGraphMode.FULL, False, MoECommType.MC2),
        (None, None, CUDAGraphMode.NONE, None, MoECommType.FUSED_MC2),
        (None, False, CUDAGraphMode.NONE, False, MoECommType.MC2),
        (True, False, CUDAGraphMode.NONE, True, MoECommType.FUSED_MC2),
    ],
)
def test_platform_passes_authoritative_phase_and_actual_backend(
    metadata_phase, profile_phase, mode, expected_phase, comm
):
    metadata = None if metadata_phase is None else AscendAttentionMetadataMap(has_prefill=metadata_phase)
    config = SimpleNamespace(use_v2_model_runner=True, parallel_config=SimpleNamespace(prefill_context_parallel_size=1))
    selector = Mock(return_value=comm)
    with (
        patch("vllm_ascend.quantization.utils.get_dynamic_mx_quant_scale_alg", return_value=0),
        patch("vllm_ascend.platform.is_moe_model", return_value=True),
        patch("vllm.distributed.get_tensor_model_parallel_world_size", return_value=8),
        patch("vllm.distributed.get_dp_group", return_value=SimpleNamespace(world_size=1)),
        patch("vllm_ascend.ascend_forward_context.select_moe_comm_method", selector),
        patch("vllm_ascend.ascend_forward_context.use_cann_megamoe", return_value=True),
        patch("vllm_ascend.ascend_forward_context._is_decode_only_node", return_value=False),
        patch("vllm_ascend.ascend_forward_context.get_mc2_mask", return_value=None),
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_moe_comm_method", return_value=object()),
        override_mrv2_in_profile_run(profile_phase is not None, has_prefill=profile_phase),
    ):
        extra = NPUPlatform.set_additional_forward_context(
            metadata, config, None, num_tokens=8, cudagraph_runtime_mode=mode
        )
    selector.assert_called_once_with(8, config, has_prefill=expected_phase)
    assert extra["use_mega_moe"] is (comm is MoECommType.FUSED_MC2)


@pytest.mark.parametrize("disable_for_decode,mega_enabled", [(True, True), (False, True), (True, False)])
def test_platform_rejects_lost_phase_only_when_phase_routing_is_active(disable_for_decode, mega_enabled):
    config = SimpleNamespace(use_v2_model_runner=True, parallel_config=SimpleNamespace(prefill_context_parallel_size=1))
    selector = Mock(return_value=MoECommType.FUSED_MC2)
    with (
        patch("vllm_ascend.quantization.utils.get_dynamic_mx_quant_scale_alg", return_value=0),
        patch("vllm_ascend.platform.is_moe_model", return_value=True),
        patch(
            "vllm_ascend.platform.get_ascend_config",
            return_value=SimpleNamespace(disable_megamoe_for_decode=disable_for_decode),
        ),
        patch("vllm.distributed.get_tensor_model_parallel_world_size", return_value=8),
        patch("vllm.distributed.get_dp_group", return_value=SimpleNamespace(world_size=1)),
        patch("vllm_ascend.ascend_forward_context.select_moe_comm_method", selector),
        patch("vllm_ascend.ascend_forward_context.use_cann_megamoe", return_value=mega_enabled),
        patch("vllm_ascend.ascend_forward_context._is_decode_only_node", return_value=False),
        patch("vllm_ascend.ascend_forward_context.get_mc2_mask", return_value=None),
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_moe_comm_method", return_value=object()),
    ):
        if disable_for_decode and mega_enabled:
            with pytest.raises(RuntimeError, match="scheduler's has_prefill"):
                NPUPlatform.set_additional_forward_context({}, config, None, num_tokens=8)
            selector.assert_not_called()
        else:
            NPUPlatform.set_additional_forward_context({}, config, None, num_tokens=8)
            selector.assert_called_once_with(8, config, has_prefill=None)


@pytest.mark.parametrize(
    "decode_tokens,staged_capacity,expected",
    [(128, 4096, [128]), (8192, 4096, [4096, 8192])],
)
@pytest.mark.parametrize("mega_enabled", [True, False])
def test_memory_profile_reserves_staged_before_mega_and_restores_phase(
    decode_tokens, staged_capacity, expected, mega_enabled
):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.ascend_config = SimpleNamespace(
        disable_megamoe_for_decode=True,
        sparse_kv_offload_config=SimpleNamespace(enabled=False),
    )
    runner.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(prefill_context_parallel_size=1))
    runner.compilation_config = SimpleNamespace(
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY, max_cudagraph_capture_size=decode_tokens
    )
    runner.max_num_tokens = 8192
    runner.max_num_reqs = decode_tokens
    runner.decode_query_len = 1
    runner.get_model = Mock(return_value=object())
    phases = []

    def record_staged(num_tokens, **kwargs):
        assert kwargs == dict(skip_attn=True, skip_eplb=True, uniform_decode=True, is_profile=True)
        assert get_mrv2_in_profile_run() is True
        phases.append((num_tokens, get_mrv2_profile_has_prefill()))

    def record_parent(_self):
        assert get_mrv2_in_profile_run() is True
        phases.append((8192, get_mrv2_profile_has_prefill()))

    runner._dummy_run = Mock(side_effect=record_staged)
    with (
        patch("vllm_ascend.worker.v2.model_runner.get_mc2_tokens_capacity", return_value=8192),
        patch("vllm_ascend.worker.v2.model_runner.get_staged_mc2_tokens_capacity", return_value=staged_capacity),
        patch("vllm_ascend.worker.v2.model_runner.use_cann_megamoe", return_value=mega_enabled),
        patch("vllm_ascend.worker.v2.model_runner.disable_compilation", return_value=nullcontext()),
        patch.object(GPUModelRunner, "profile_run", record_parent),
    ):
        runner.profile_run()
    expected_staged = [(tokens, False) for tokens in expected] if mega_enabled else []
    assert phases == expected_staged + [(8192, None)]
    assert get_mrv2_in_profile_run() is False
    assert get_mrv2_profile_has_prefill() is None
