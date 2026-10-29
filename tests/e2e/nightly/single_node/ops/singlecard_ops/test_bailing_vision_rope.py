# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config

from vllm_ascend.ops.rotary_embedding import AscendApplyRotaryEmb

HEAD_DIM = 72
NUM_LOCAL_HEADS = 2  # The checkpoint has 16 vision heads, with TP=8.
GRID_HEIGHT = 4
GRID_WIDTH = 6
MERGE_SIZE = 2
ROPE_THETA = 10000.0


@pytest.fixture(autouse=True)
def default_vllm_config():
    with set_current_vllm_config(VllmConfig()):
        yield


@pytest.mark.parametrize("layout", ["packed_qk", "unbatched"])
@torch.inference_mode()
def test_bailing_vision_rope_fp32_compute_preserves_bf16_head_dim_72(layout):
    """Exercise the real NPU FP32 rotary operator used by the BF16 vision tower."""
    torch.npu.set_device(0)
    rotary = AscendApplyRotaryEmb(enforce_enable=True, is_neox_style=True, enable_fp32_compute=True)
    assert rotary.enable_fp32_compute

    # Independently reproduce HF's spatial-merge order and FP32 frequencies.
    # Keep the reference on CPU so it cannot accidentally reuse the NPU op.
    coordinates = torch.tensor(
        [
            [block_h + local_h, block_w + local_w]
            for block_h in range(0, GRID_HEIGHT, MERGE_SIZE)
            for block_w in range(0, GRID_WIDTH, MERGE_SIZE)
            for local_h in range(MERGE_SIZE)
            for local_w in range(MERGE_SIZE)
        ],
        dtype=torch.float32,
        device="cpu",
    )
    rotary_dim = HEAD_DIM // 2
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device="cpu") / rotary_dim))
    frequencies = (coordinates[..., None] * inv_freq).flatten(1)
    cos_cpu, sin_cpu = frequencies.cos(), frequencies.sin()
    cos, sin = cos_cpu.to("npu"), sin_cpu.to("npu")
    assert cos.dtype == sin.dtype == torch.float32

    generator = torch.Generator(device="cpu").manual_seed(3)
    query_key_cpu = torch.randn(
        2, GRID_HEIGHT * GRID_WIDTH, NUM_LOCAL_HEADS, HEAD_DIM, generator=generator, dtype=torch.bfloat16, device="cpu"
    )
    if layout == "unbatched":
        query_key_cpu = query_key_cpu[0]
    query_key = query_key_cpu.to("npu")

    # This calls torch_npu.npu_rotary_mul with real FP32 inputs after the
    # module's promotion; unsupported hardware/dtype combinations must fail.
    actual = rotary.forward_oot(query_key, cos, sin)

    qk_float = query_key_cpu.float()
    rotate_half = torch.cat((-qk_float[..., HEAD_DIM // 2 :], qk_float[..., : HEAD_DIM // 2]), dim=-1)
    cos_full = torch.cat((cos_cpu, cos_cpu), dim=-1).unsqueeze(1)
    sin_full = torch.cat((sin_cpu, sin_cpu), dim=-1).unsqueeze(1)
    expected = (qk_float * cos_full + rotate_half * sin_full).to(torch.bfloat16)

    assert rotary.enable_fp32_compute
    assert actual.dtype == query_key.dtype == torch.bfloat16
    assert actual.shape == query_key.shape
    assert actual.device == query_key.device
    assert cos.dtype == sin.dtype == torch.float32
    # A fused FP32 multiply-add may cross a final BF16 rounding midpoint.
    torch.testing.assert_close(actual.cpu(), expected, rtol=torch.finfo(torch.bfloat16).eps, atol=1e-5)
    torch.testing.assert_close(query_key.cpu(), query_key_cpu, rtol=0, atol=0)
    torch.testing.assert_close(cos.cpu(), cos_cpu, rtol=0, atol=0)
    torch.testing.assert_close(sin.cpu(), sin_cpu, rtol=0, atol=0)
