# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu.model_states.default import DefaultModelState

from vllm_ascend.attention.mla_v1 import AscendMLAMetadataBuilder
from vllm_ascend.worker.v2 import attn_utils
from vllm_ascend.worker.v2.model_states.default import AscendModelState
from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import AscendAutoRegressiveSpeculator


def test_mla_mrope_prepares_current_batch_before_metadata_and_keeps_absolute_positions():
    state = AscendModelState.__new__(AscendModelState)
    state.model_config = SimpleNamespace(use_mla=True)
    positions = torch.full((3, 7), 99, dtype=torch.long)
    state.rope_state = SimpleNamespace(positions=positions, get_positions=lambda n: positions[:, :n])
    state.prepare_engram_inputs = lambda *_: {}
    absolute = torch.tensor([10, 11, 20, 21, 0, 0])
    batch = SimpleNamespace(
        positions=absolute,
        idx_mapping=torch.tensor([2, 0]),
        query_start_loc=torch.tensor([0, 2, 4]),
        num_tokens=4,
        num_tokens_after_padding=6,
    )
    requests = SimpleNamespace(
        prefill_len=SimpleNamespace(gpu=torch.tensor([16, 8, 24])),
        num_computed_tokens=SimpleNamespace(gpu=torch.tensor([20, 0, 10])),
    )
    original_absolute = absolute.clone()
    address = positions.data_ptr()
    for coordinates in (
        torch.tensor([[10, 11, 20, 21], [-2, 0, 1, 2], [2, 1, 0, -2]]),
        torch.tensor([[30, 31, 40, 41], [1, -2, 3, 0], [-3, 2, 0, 1]]),
    ):
        state.rope_state.prepare_positions = Mock(
            side_effect=lambda *_, current=coordinates: positions[:, :4].copy_(current)
        )
        state.prepare_attn_rotary_positions(batch, requests)
        metadata_positions = state.get_attn_rotary_positions(6)
        model_positions = state.prepare_inputs(batch, requests)["positions"]
        torch.testing.assert_close(metadata_positions[:, :4], coordinates, rtol=0, atol=0)
        assert torch.count_nonzero(metadata_positions[:, 4:]) == 0
        assert model_positions.data_ptr() == metadata_positions.data_ptr() == address
        state.rope_state.prepare_positions.assert_called_once_with(
            batch.idx_mapping, batch.query_start_loc, requests.prefill_len.gpu, requests.num_computed_tokens.gpu
        )
        torch.testing.assert_close(absolute, original_absolute, rtol=0, atol=0)


@pytest.mark.parametrize("use_mla,has_rope", [(False, True), (True, False)])
def test_other_models_keep_upstream_position_preparation(use_mla, has_rope):
    state = AscendModelState.__new__(AscendModelState)
    state.model_config = SimpleNamespace(use_mla=use_mla)
    state.rope_state = SimpleNamespace(prepare_positions=Mock()) if has_rope else None
    state.prepare_engram_inputs = lambda *_: {}
    batch, requests = object(), object()
    state.prepare_attn_rotary_positions(batch, requests)
    assert state.get_attn_rotary_positions(4) is None
    if has_rope:
        state.rope_state.prepare_positions.assert_not_called()
    with patch.object(DefaultModelState, "prepare_inputs", return_value={"upstream": True}) as upstream:
        assert state.prepare_inputs(batch, requests) == {"upstream": True}
        upstream.assert_called_once_with(batch, requests)


@pytest.mark.parametrize("capture", [False, True])
def test_only_bailing_mla_groups_receive_three_axis_positions(capture):
    bailing = AscendMLAMetadataBuilder.__new__(AscendMLAMetadataBuilder)
    bailing.bailing_rotary_emb = object()
    ordinary = AscendMLAMetadataBuilder.__new__(AscendMLAMetadataBuilder)
    ordinary.bailing_rotary_emb = None
    other = SimpleNamespace()
    groups = []
    for name, builder in (("target", bailing), ("ordinary", ordinary), ("gdn", other)):
        builder.build = lambda common_prefix_len, common_attn_metadata, **_: common_attn_metadata
        builder.build_for_cudagraph_capture = lambda common, **_: common
        # The draft shares the same actual Bailing RoPE, so step zero must
        # retain the complete target coordinates including image tokens.
        names = [name, "draft"] if name == "target" else [name]
        groups.append(SimpleNamespace(layer_names=names, get_metadata_builder=lambda _, b=builder: b))
    absolute = torch.tensor([40, 41, 60, 61])
    rotary = torch.tensor([[4, 5, 8, 9], [-2, 0, 3, -1], [2, -3, 0, 1]])
    slots = torch.arange(4).reshape(1, 4)
    with patch.object(attn_utils, "uses_turboquant_groups", return_value=False):
        metadata = attn_utils.build_attn_metadata(
            attn_groups=[groups],
            num_reqs=2,
            num_tokens=4,
            query_start_loc_gpu=torch.tensor([0, 2, 4], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 2, 4], dtype=torch.int32),
            max_query_len=2,
            seq_lens=torch.tensor([42, 62], dtype=torch.int32),
            max_seq_len=128,
            block_tables=[torch.zeros((2, 2), dtype=torch.int32)],
            slot_mappings=slots,
            kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace()]),
            seq_lens_np=np.array([42, 62], dtype=np.int32),
            positions=absolute,
            rotary_positions=rotary,
            for_cudagraph_capture=capture,
            skip_ring_state_update=False,
        )
    assert metadata["target"] is metadata["draft"]
    assert metadata["target"].positions is rotary
    assert metadata["ordinary"].positions is metadata["gdn"].positions is absolute
    for common in metadata.values():
        torch.testing.assert_close(common.slot_mapping, slots[0])


def test_mtp_delta_reorder_replaces_source_without_changing_graph_buffer_or_slots():
    spec = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    spec._mrope_decode_delta = torch.zeros(4, dtype=torch.long)
    spec.model_state = SimpleNamespace(rope_state=SimpleNamespace(prefill_delta=SimpleNamespace(gpu=None)))
    absolute = torch.tensor([40, 60, 0, 0])
    spec.input_buffers = SimpleNamespace(positions=absolute)
    original = absolute.clone()
    buffer_address = spec._mrope_decode_delta.data_ptr()
    for source, order in (([3, -2, -7], [2, 0]), ([-8, 6, 2], [1, 2])):
        spec.model_state.rope_state.prefill_delta.gpu = torch.tensor(source)
        batch = SimpleNamespace(num_reqs=2, idx_mapping=torch.tensor(order))
        spec._prepare_mrope_decode_delta(batch, dummy_run=False)
        expected_delta = torch.tensor([source[i] for i in order] + [0, 0])
        assert spec._mrope_decode_delta.data_ptr() == buffer_address
        torch.testing.assert_close(spec._mrope_decode_delta, expected_delta, rtol=0, atol=0)
        assert spec._get_mrope_decode_positions(4, 0) is None
        for step in (1, 2):
            torch.testing.assert_close(spec._get_mrope_decode_positions(4, step), absolute + expected_delta)
        torch.testing.assert_close(absolute, original, rtol=0, atol=0)
    spec._prepare_mrope_decode_delta(batch, dummy_run=True)
    assert torch.count_nonzero(spec._mrope_decode_delta) == 0
    with pytest.raises(ValueError, match="one token per request"):
        spec._get_mrope_decode_positions(5, 1)


def test_non_mrope_draft_does_not_prepare_or_transform_positions():
    spec = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    assert spec._mrope_decode_delta is None
    spec._prepare_mrope_decode_delta(object(), dummy_run=False)
    assert spec._get_mrope_decode_positions(128, 1) is None
