# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch


@pytest.fixture
def host_lengths(monkeypatch):
    from vllm.model_executor.models import bailing_moe_v3_vl

    from vllm_ascend.ops import mm_encoder_attention

    class VisionTransformer:
        def prepare_encoder_metadata(self, grid_thw_list, **kwargs):
            pass

    class EncoderAttention:
        @classmethod
        def maybe_recompute_cu_seqlens(cls, *args, **kwargs):
            pass

    monkeypatch.setattr(bailing_moe_v3_vl, "BailingMoeV3VisionTransformer", VisionTransformer)
    monkeypatch.setattr(mm_encoder_attention, "AscendMMEncoderAttention", EncoderAttention)
    patch_path = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/worker/patch_bailing_vit_cpu_lengths.py"
    spec = spec_from_file_location("_test_bailing_vit_cpu_lengths", patch_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "get_encoder_forward_context", lambda: SimpleNamespace(capturing=False))
    return module


@pytest.mark.parametrize("endpoints", [[0, 4056], [0, 4056, 8112], [0, 8192, 16384, 32768]])
def test_eager_bailing_preserves_cpu_lengths(host_lengths, monkeypatch, endpoints):
    backend = host_lengths.AttentionBackendEnum.TORCH_SDPA
    vision = object.__new__(host_lengths.BailingMoeV3VisionTransformer)
    vision._use_host_cu_seqlens = True
    vision.dtype = torch.bfloat16
    vision.attn_backend = backend
    lengths = np.array(endpoints, dtype=np.int32)
    attention = host_lengths.AscendMMEncoderAttention

    def prepare(*args, **kwargs):
        return attention.maybe_recompute_cu_seqlens(backend, lengths, 1152, 8, SimpleNamespace(type="npu"))

    monkeypatch.setattr(host_lengths, "_original_prepare_encoder_metadata", prepare)
    actual = vision.prepare_encoder_metadata([])

    assert actual.device.type == "cpu"
    assert actual.dtype == torch.int32
    assert actual.tolist() == endpoints
    assert not host_lengths._bailing_vision_metadata_scope.get()


def test_graph_and_exception_restore_original_behavior(host_lengths, monkeypatch):
    vision = object.__new__(host_lengths.BailingMoeV3VisionTransformer)
    vision._use_host_cu_seqlens = True
    vision.dtype = torch.bfloat16
    vision.attn_backend = host_lengths.AttentionBackendEnum.TORCH_SDPA
    original = Mock(side_effect=RuntimeError("encoder metadata error"))
    monkeypatch.setattr(host_lengths, "_original_prepare_encoder_metadata", original)

    with pytest.raises(RuntimeError, match="encoder metadata error"):
        vision.prepare_encoder_metadata([])
    assert not host_lengths._bailing_vision_metadata_scope.get()

    original.side_effect = lambda *args, **kwargs: host_lengths._bailing_vision_metadata_scope.get()
    assert not vision.prepare_encoder_metadata([], max_batch_size=1)
    assert not vision.prepare_encoder_metadata([], max_frames_per_batch=1)
    assert not vision.prepare_encoder_metadata([], max_seqlen_override=1)

    fallback = Mock()
    monkeypatch.setattr(host_lengths, "_original_recompute_cu_seqlens", fallback)
    assert (
        host_lengths.AscendMMEncoderAttention.maybe_recompute_cu_seqlens(
            vision.attn_backend, np.array([0, 4056], dtype=np.int32), 1152, 8, SimpleNamespace(type="npu")
        )
        is fallback.return_value
    )
