import pytest
import torch

from vllm_ascend.ops.triton.activation.swiglustep import swiglustep_forward_triton
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def _swiglustep_reference(x: torch.Tensor, limit: float = 7.0) -> torch.Tensor:
    # Independent of the kernel under test: slice gate (first half) and up
    # (second half), apply silu-then-clamp(max) on gate and symmetric clamp
    # on up -- the SwigluStep order, not clamp-then-silu (SwigluOAI).
    d = x.shape[-1] // 2
    gate = torch.nn.functional.silu(x[..., :d]).clamp(max=limit)
    up = x[..., d:].clamp(min=-limit, max=limit)
    return gate * up


# The kernel requires N (= last_dim // 2) to be a multiple of 16 for
# bf16/fp16 (32-byte UB alignment on NPU vector core), so every shape here
# has a last dim that is a multiple of 32. Real MoE shape is N=1280
# (Step-3.7 moe_intermediate_size).
@pytest.mark.parametrize(
    ("shape", "dtype", "limit"),
    [
        ((1, 2560), torch.float16, 7.0),
        ((128, 2560), torch.float16, 7.0),
        ((4000, 2560), torch.bfloat16, 7.0),
        ((4, 128, 2560), torch.bfloat16, 7.0),
        ((8, 4096), torch.float16, 1.0),
    ],
)
@torch.inference_mode()
def test_swiglustep_triton_correctness(shape, dtype, limit):
    """compare swiglustep_forward_triton with an independent pytorch baseline."""
    init_device_properties_triton()
    device = "npu"

    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=dtype, device=device)

    out_triton = swiglustep_forward_triton(x, limit=limit)
    out_ref = _swiglustep_reference(x, limit=limit)

    # bf16 loses ~2 digits around silu(large) / clamp boundary; fp16 is tighter.
    rtol, atol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (5e-3, 5e-3)

    assert out_triton.shape == out_ref.shape
    assert out_triton.dtype == out_ref.dtype
    assert torch.allclose(out_triton, out_ref, rtol=rtol, atol=atol)


@torch.inference_mode()
def test_swiglustep_triton_clamps_to_golden_value():
    """gate=+100 -> silu(~100) clamped to limit(7); up=-100 -> -7; out = -49."""
    init_device_properties_triton()
    # N=16 to satisfy N%16==0 (32-byte UB alignment for fp16)
    x = torch.full((1, 32), 100.0, dtype=torch.float16, device="npu")
    x[:, 16:] = -100.0  # up half = second half of columns

    out = swiglustep_forward_triton(x, limit=7.0)

    assert out.shape == (1, 16)
    expected = torch.full((1, 16), -49.0, dtype=torch.float16, device="npu")
    assert torch.allclose(out, expected, atol=1e-3)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


@pytest.mark.parametrize("limit", [4.0, 10.0])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_swiglustep_triton_configured_limit_boundaries(limit, dtype):
    """Configured bounds preserve SiLU-before-clamp, signs and input storage."""
    init_device_properties_triton()
    gate = torch.tensor(
        [
            -100.0,
            -20.0,
            -2.0,
            -0.5,
            0.0,
            1.0,
            limit - 0.125,
            limit,
            limit + 0.125,
            limit + 1.0,
            7.0,
            8.0,
            10.0,
            12.0,
            20.0,
            100.0,
        ],
        dtype=dtype,
    )
    up = torch.tensor(
        [
            -100.0,
            -10.0,
            -4.0,
            -1.0,
            0.0,
            0.5,
            1.0,
            limit,
            limit + 0.125,
            -limit - 0.125,
            10.0,
            7.0,
            4.0,
            2.0,
            100.0,
            -100.0,
        ],
        dtype=dtype,
    )
    cpu_input = torch.cat((gate, up)).unsqueeze(0)
    x = cpu_input.to("npu")
    # Kernel arithmetic is FP32 with one final cast; avoid a low-precision
    # reference that rounds between SiLU, clamp and multiplication.
    expected = _swiglustep_reference(cpu_input.float(), limit=limit).to(dtype)
    actual = swiglustep_forward_triton(x, limit=limit).cpu()
    rtol, atol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (5e-3, 5e-3)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(x.cpu(), cpu_input, rtol=0, atol=0)
    assert actual[0, -1].item() == -(limit * limit)


@torch.inference_mode()
def test_swiglustep_triton_handles_non_contiguous_input():
    """the wrapper must contiguous() a strided view before launching."""
    init_device_properties_triton()
    torch.manual_seed(0)
    base = torch.randn(4, 5120, dtype=torch.bfloat16, device="npu")
    x = base[:, ::2]  # (4, 2560) strided view, non-contiguous
    assert not x.is_contiguous()

    out_triton = swiglustep_forward_triton(x, limit=7.0)
    out_ref = _swiglustep_reference(x, limit=7.0)

    assert out_triton.shape == out_ref.shape
    assert torch.allclose(out_triton, out_ref, atol=2e-2, rtol=2e-2)
