# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.attention.mla_v1 import AscendMLAImpl, AscendMLAMetadataBuilder, _get_bailing_rotary_embedding


def test_bailing_group_binds_actual_rotary_even_for_text_mtp_config():
    rope = SimpleNamespace(uses_mla_mrope=True)
    layers = {name: SimpleNamespace(impl=SimpleNamespace(rotary_emb=rope)) for name in ("target", "draft")}
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="bailing_hybrid_v3_mtp")),
        compilation_config=SimpleNamespace(static_forward_context=layers),
    )
    assert _get_bailing_rotary_embedding(config, ["target", "draft"]) is rope
    layers["draft"].impl.rotary_emb = SimpleNamespace(uses_mla_mrope=False)
    with pytest.raises(ValueError, match="must share"):
        _get_bailing_rotary_embedding(config, ["target", "draft"])
    assert _get_bailing_rotary_embedding(config, ["draft"]) is None


def _builder():
    builder = AscendMLAMetadataBuilder.__new__(AscendMLAMetadataBuilder)
    builder.use_mla_rope = True
    builder.num_decodes = 1
    builder.num_decode_tokens = 4
    builder.num_actual_tokens = 9
    builder.query_lens = torch.tensor([4, 2, 3])
    builder.seq_lens = torch.tensor([40, 2, 3])
    builder.block_table = torch.zeros(3, 2, dtype=torch.int32)
    builder.attn_mask_builder = SimpleNamespace(get_splitfuse_attn_mask=lambda: None)
    builder.build_chunked_metadata = lambda *_: None
    builder.nope_zero_rope_cache = None
    builder._get_cos_and_sin = Mock(return_value=(torch.empty(12, 1, 1, 64), torch.empty(12, 1, 1, 64)))
    return builder


@pytest.mark.parametrize("uses_rope", [True, False])
def test_mla_prefill_three_axes_slice_tokens_and_keep_padding(uses_rope):
    builder = _builder()
    builder.use_mla_rope = uses_rope
    positions = torch.arange(36).reshape(3, 12)
    common = SimpleNamespace(positions=positions, query_start_loc=torch.tensor([0, 4, 6, 9]))
    metadata = builder.build_prefill_metadata(0, common)
    torch.testing.assert_close(metadata.input_positions, positions[:, 4:])
    assert metadata.input_positions.shape == (3, 8)  # Five real tokens and three padding rows.
    if uses_rope:
        torch.testing.assert_close(builder._get_cos_and_sin.call_args.args[0], positions[:, 4:])
    else:
        builder._get_cos_and_sin.assert_not_called()
        assert metadata.cos is metadata.sin is None


@pytest.mark.parametrize("three_axes", [True, False])
def test_mla_decode_mtp_graph_padding_uses_token_axis(three_axes):
    builder = _builder()
    builder.num_decodes = 2
    builder.num_decode_tokens = builder.num_actual_tokens = 8
    builder.query_lens = torch.tensor([4, 4])
    builder.seq_lens = torch.tensor([40, 60])
    builder.graph_pad_size = 12
    builder.slot_mapping = torch.arange(8)
    builder.block_table = torch.zeros(2, 2, dtype=torch.int32)
    builder.speculative_config = SimpleNamespace(disable_padded_drafter_batch=False)
    builder.get_block_table_size = lambda *_: 2
    builder.pad_actual_seq_len_q_mtp_enable_pad = lambda *_: [4, 8, 12]
    positions = torch.arange(24).reshape(3, 8)
    if not three_axes:
        positions = positions[0]
    common = SimpleNamespace(
        positions=positions,
        num_reqs=2,
        query_start_loc_cpu=torch.tensor([0, 4, 8]),
        decode_token_per_req=4,
    )
    metadata = builder.build_decode_metadata(0, common)
    expected = torch.cat((positions, torch.zeros((*positions.shape[:-1], 4), dtype=positions.dtype)), dim=-1)
    torch.testing.assert_close(metadata.input_positions, expected)
    torch.testing.assert_close(builder._get_cos_and_sin.call_args.args[0], expected)
    assert builder._get_cos_and_sin.call_args.kwargs == {"use_cache": True}
    assert metadata.cos.shape[0] == 8


def test_bailing_mrope_rejects_flashmla_without_changing_other_modes():
    impl = AscendMLAImpl.__new__(AscendMLAImpl)
    impl.rotary_emb = SimpleNamespace(uses_mla_mrope=True)
    # Even an otherwise-uninitialized implementation must reject the 1D-only backend.
    assert not impl._can_use_flashmla()
