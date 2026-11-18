# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.model_loader.reload.layerwise import _finalize_attention_layer
from vllm.model_executor.model_loader.reload.types import LayerReloadingInfo
from vllm.model_executor.model_loader.reload.utils import get_layer_params_buffers
from vllm.model_executor.models.bailing_moe_v3 import (
    BailingMoeV3KimiDeltaAttention,
    bailing_v3_kda_attention,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID

from tests.ut.ops.test_gdn_attn_builder import (
    BatchSpec,
    _build_attn_metadata,
    _patch_missing_runtime_cdiv,
)
from vllm_ascend.ops import kda_state_copy_plan
from vllm_ascend.ops.bailing_moe_v3_kda import (
    AscendBailingMoeV3KimiDeltaAttention,
    _contiguous_packed_qkv,
    _native_decode_prepare_compatible,
    _zero_padded_spec_output,
)


class _FixedLinear(nn.Module):
    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.value = value

    def forward(self, _input: torch.Tensor):
        return self.value, None


class _RecordingNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: tuple[torch.Tensor, torch.Tensor] | None = None
        self.out: torch.Tensor | None = None

    def forward_oot(self, value: torch.Tensor, gate: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
        self.inputs = (value, gate)
        self.out = out
        return value if out is None else out


def _new_attention() -> AscendBailingMoeV3KimiDeltaAttention:
    attention = AscendBailingMoeV3KimiDeltaAttention.__new__(AscendBailingMoeV3KimiDeltaAttention)
    nn.Module.__init__(attention)
    attention._ascend_kda_state_copy = None
    return attention


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("num_tokens", [1, 4])
@pytest.mark.parametrize("beta_width", [0, 2], ids=["qkv", "qkvb"])
def test_packed_qkv_reuses_only_contiguous_projection_storage(dtype, num_tokens, beta_width):
    width = 3
    projection = torch.arange(num_tokens * (3 * width + beta_width), dtype=dtype).reshape(num_tokens, -1)
    q, k, v, _beta = projection.split([width, width, width, beta_width], dim=-1)
    original = projection.clone()
    expected = torch.cat((q, k, v), dim=-1)
    can_alias = num_tokens == 1 or beta_width == 0

    if can_alias:
        with patch(
            "vllm_ascend.ops.bailing_moe_v3_kda.torch.cat",
            side_effect=AssertionError("adjacent contiguous slices must not be copied"),
        ):
            actual = _contiguous_packed_qkv(q, k, v)
        assert actual.untyped_storage().data_ptr() == projection.untyped_storage().data_ptr()
        assert actual.storage_offset() == q.storage_offset()
    else:
        with patch("vllm_ascend.ops.bailing_moe_v3_kda.torch.cat", wraps=torch.cat) as cat:
            actual = _contiguous_packed_qkv(q, k, v)
        cat.assert_called_once()
        assert actual.untyped_storage().data_ptr() != projection.untyped_storage().data_ptr()

    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(projection, original, rtol=0, atol=0)


def test_packed_qkv_preserves_nonzero_storage_offset():
    projection = torch.arange(30, dtype=torch.float32).reshape(2, 15)
    q, k, v = projection[1:2, 2:11].split(3, dim=-1)
    expected = torch.cat((q, k, v), dim=-1)

    with patch(
        "vllm_ascend.ops.bailing_moe_v3_kda.torch.cat",
        side_effect=AssertionError("the single QKV row is contiguous at its original offset"),
    ):
        actual = _contiguous_packed_qkv(q, k, v)

    assert actual.untyped_storage().data_ptr() == projection.untyped_storage().data_ptr()
    assert actual.storage_offset() == 17
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["separate_storage", "non_adjacent", "strided_columns", "different_dtype"])
def test_packed_qkv_falls_back_for_unproven_layouts(layout):
    projection = torch.arange(72, dtype=torch.float32).reshape(2, 36)
    if layout == "separate_storage":
        q, k, v = (projection[:, start : start + 3].clone() for start in (0, 3, 6))
    elif layout == "non_adjacent":
        q, k, v = (projection[:, start : start + 3] for start in (0, 4, 8))
    elif layout == "strided_columns":
        q, k, v = (projection[:, start : start + 6 : 2] for start in (0, 6, 12))
    else:
        q, k, v = projection[:, :9].split(3, dim=-1)
        k = k.to(torch.bfloat16)
    expected = torch.cat((q, k, v), dim=-1)

    with patch("vllm_ascend.ops.bailing_moe_v3_kda.torch.cat", wraps=torch.cat) as cat:
        actual = _contiguous_packed_qkv(q, k, v)

    cat.assert_called_once()
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_packed_qkv_accepts_empty_graph_token_slices():
    projection = torch.empty(0, 11)
    q, k, v, _beta = projection.split([3, 3, 3, 2], dim=-1)

    actual = _contiguous_packed_qkv(q, k, v)

    assert actual.shape == (0, 9)
    assert actual.dtype == projection.dtype
    assert actual.is_contiguous()


@pytest.mark.parametrize("shared_projection", [False, True], ids=["independent", "packed"])
def test_packed_qkv_fake_tensors_use_cat_without_storage_pointer_access(shared_projection):
    with FakeTensorMode():
        if shared_projection:
            q, k, v = torch.empty(1, 9, device="cpu").split(3, dim=-1)
        else:
            q, k, v = (torch.empty(1, 3, device="cpu") for _ in range(3))
        with (
            patch.object(
                torch.UntypedStorage,
                "data_ptr",
                side_effect=AssertionError("FakeTensor storage pointers cannot establish aliasing"),
            ),
            patch("vllm_ascend.ops.bailing_moe_v3_kda.torch.cat", wraps=torch.cat) as cat,
        ):
            actual = _contiguous_packed_qkv(q, k, v)

    cat.assert_called_once()
    assert isinstance(actual, FakeTensor)
    assert actual.shape == (1, 9)
    assert actual.device == torch.device("cpu")
    assert actual.dtype == q.dtype


@pytest.mark.parametrize("num_speculative_tokens", [0, 1, 7, 8])
def test_init_enforces_recurrent_sequence_length_limit(num_speculative_tokens):
    def fake_base_init(instance, *_args, **_kwargs):
        nn.Module.__init__(instance)
        instance.num_speculative_tokens = num_speculative_tokens
        instance.model_config = SimpleNamespace(dtype=torch.bfloat16)
        instance.conv_size = 4
        instance.projection_size_per_partition = 6
        instance.q_conv1d = nn.Module()
        instance.q_conv1d.weight = nn.Parameter(torch.empty(6, 1, 4))

    with (
        patch.object(BailingMoeV3KimiDeltaAttention, "__init__", fake_base_init),
        patch("vllm_ascend.ops.bailing_moe_v3_kda.is_conv_state_dim_first", return_value=False),
    ):
        if num_speculative_tokens <= 7:
            attention = AscendBailingMoeV3KimiDeltaAttention(None)
            assert attention._requires_kda_state_copy is True
            assert "_requires_kda_state_copy" not in attention.__dict__
            assert attention._ascend_kda_state_copy is None
        else:
            with pytest.raises(ValueError, match="at most 7 speculative tokens"):
                AscendBailingMoeV3KimiDeltaAttention(None)


def test_upstream_custom_op_calls_ascend_forward_with_g1_keyword():
    attention = _new_attention()
    attention.prefix = "model.layers.0.self_attn"
    output = torch.full((1, 2, 1, 2), torch.nan)
    qkv = torch.empty(2, 2)
    gate = torch.empty(1, 2, 1, 2)
    beta = torch.empty(1, 2, 1)

    with (
        patch(
            "vllm.model_executor.models.bailing_moe_v3.get_forward_context",
            return_value=SimpleNamespace(no_compile_layers={attention.prefix: attention}),
        ),
        patch(
            "vllm_ascend.ops.bailing_moe_v3_kda.get_forward_context",
            return_value=SimpleNamespace(attn_metadata=None),
        ),
    ):
        bailing_v3_kda_attention(qkv, qkv, qkv, gate, beta, output, attention.prefix)

    assert torch.equal(output, torch.zeros_like(output))


@pytest.mark.parametrize("separate_b_proj", [False, True])
@pytest.mark.parametrize("flash_kda_enabled", [False, True])
def test_forward_passes_raw_gate_and_backend_beta(separate_b_proj: bool, flash_kda_enabled: bool):
    attention = _new_attention()
    num_tokens, num_heads, head_dim = 4, 2, 3
    projection_size = num_heads * head_dim
    qkvb = torch.arange(
        num_tokens * (3 * projection_size + num_heads),
        dtype=torch.bfloat16,
    ).reshape(num_tokens, -1)
    expected_q, expected_k, expected_v, beta_logits = qkvb.split(
        [projection_size, projection_size, projection_size, num_heads],
        dim=-1,
    )
    raw_gate = torch.linspace(-3, 3, num_tokens * projection_size, dtype=torch.float32).reshape(
        num_tokens, projection_size
    )
    output_gate = torch.randn(num_tokens, projection_size)

    attention.separate_b_proj = separate_b_proj
    attention.qkvb_proj = None if separate_b_proj else _FixedLinear(qkvb)
    attention.qkv_proj = (
        _FixedLinear(torch.cat((expected_q, expected_k, expected_v), dim=-1)) if separate_b_proj else None
    )
    attention.b_proj = _FixedLinear(beta_logits) if separate_b_proj else None
    attention.f_proj = _FixedLinear(raw_gate)
    attention.g_proj = _FixedLinear(output_gate)
    attention.o_norm = _RecordingNorm()
    attention.o_proj = _FixedLinear(torch.randn(num_tokens, projection_size))
    attention.projection_size_per_partition = projection_size
    attention.local_num_heads = num_heads
    attention.head_dim = head_dim
    attention.prefix = "model.layers.0.linear_attn"

    hidden_states = torch.randn(num_tokens, 7, dtype=torch.bfloat16)
    output = torch.empty(num_tokens, projection_size)
    with (
        patch("vllm_ascend.ops.bailing_moe_v3_kda.FLASH_KDA_ENABLED", flash_kda_enabled),
        patch.object(torch.ops.vllm, "bailing_v3_kda_attention", create=True) as custom_op,
        patch(
            "vllm_ascend.ops.bailing_moe_v3_kda.torch.zeros",
            side_effect=AssertionError("attention result is fully defined by the custom op"),
        ),
    ):
        result = attention.forward(hidden_states, torch.arange(num_tokens), output)

    assert result is None
    custom_op.assert_called_once()
    q_arg, k_arg, v_arg, gate_arg, beta_arg, core_arg, prefix_arg = custom_op.call_args.args
    torch.testing.assert_close(q_arg, expected_q)
    torch.testing.assert_close(k_arg, expected_k)
    torch.testing.assert_close(v_arg, expected_v)
    torch.testing.assert_close(gate_arg, raw_gate.reshape(1, num_tokens, num_heads, head_dim))
    expected_beta = beta_logits if flash_kda_enabled else beta_logits.float().sigmoid()
    assert beta_arg.dtype == expected_beta.dtype
    torch.testing.assert_close(beta_arg, expected_beta.unsqueeze(0))
    assert tuple(core_arg.shape) == (1, num_tokens, num_heads, head_dim)
    assert prefix_arg == attention.prefix
    assert attention.o_norm.inputs is not None
    assert attention.o_norm.inputs[0] is core_arg
    assert attention.o_norm.out is core_arg
    torch.testing.assert_close(attention.o_norm.inputs[1], output_gate.reshape(num_tokens, num_heads, head_dim))


@pytest.mark.parametrize(
    ("safe_gate", "expected_lower_bound"),
    [(False, None), (True, -4.0)],
)
@pytest.mark.parametrize("flash_kda_enabled", [False, True])
def test_recurrent_uses_current_kda_contract(
    safe_gate: bool, expected_lower_bound: float | None, flash_kda_enabled: bool
):
    attention = _new_attention()
    attention.safe_gate = safe_gate
    attention.lower_bound = -4.0
    attention.A_log = torch.randn(1)
    attention.dt_bias = torch.randn(2)
    q = torch.randn(1, 3, 1, 2)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    raw_gate = torch.randn_like(q)
    beta_logits = torch.tensor([-8.0, 0.0, 8.0], dtype=torch.bfloat16).reshape(1, 3, 1)
    beta = beta_logits if flash_kda_enabled else beta_logits.float().sigmoid()
    recurrent_state = torch.zeros(4, 1, 2, 2)
    cu_seqlens = torch.tensor([0, 3], dtype=torch.int32)
    state_indices = torch.tensor([2], dtype=torch.int32)
    accepted = torch.tensor([2], dtype=torch.int32)

    with (
        patch("vllm_ascend.ops.bailing_moe_v3_kda.FLASH_KDA_ENABLED", flash_kda_enabled),
        patch("vllm_ascend.ops.bailing_moe_v3_kda.run_recurrent_kda", return_value=v) as run_recurrent,
    ):
        actual = attention._run_recurrent(
            q,
            k,
            v,
            raw_gate,
            beta,
            recurrent_state,
            cu_seqlens,
            state_indices,
            num_accepted_tokens=accepted,
        )

    assert actual is v
    expected_args = (
        q,
        k,
        v,
        raw_gate,
        beta,
        recurrent_state,
        cu_seqlens,
        state_indices,
        attention.A_log,
        attention.dt_bias,
    )
    assert len(run_recurrent.call_args.args) == len(expected_args)
    for index, (actual_arg, expected_arg) in enumerate(zip(run_recurrent.call_args.args, expected_args)):
        if index == 4:
            assert actual_arg.dtype == torch.float32
            torch.testing.assert_close(actual_arg, beta_logits.float().sigmoid())
        else:
            assert actual_arg is expected_arg
    assert run_recurrent.call_args.kwargs["lower_bound"] == expected_lower_bound
    assert run_recurrent.call_args.kwargs["num_accepted_tokens"] is accepted


@pytest.mark.parametrize(
    ("safe_gate", "expected_lower_bound"),
    [(False, None), (True, -4.0)],
)
@pytest.mark.parametrize("has_empty_sequence", [False, True])
@pytest.mark.parametrize("cache_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("supports_fused_copy", [False, True], ids=["a3_fallback", "a5_fused"])
def test_prefill_preserves_padded_state_on_a3_and_a5(
    safe_gate: bool,
    expected_lower_bound: float | None,
    has_empty_sequence: bool,
    cache_dtype: torch.dtype,
    supports_fused_copy: bool,
):
    attention = _new_attention()
    attention.safe_gate = safe_gate
    attention.lower_bound = -4.0
    attention.A_log = torch.randn(1)
    attention.dt_bias = torch.randn(2)
    q = torch.randn(1, 3, 1, 2)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    raw_gate = torch.randn_like(q)
    beta = torch.rand(1, 3, 1)
    backing = torch.arange(36, dtype=cache_dtype).reshape(9, 1, 2, 2)
    recurrent_state = backing[1::2]
    original_backing = backing.clone()
    original_state = recurrent_state.clone()
    assert recurrent_state.storage_offset() > 0
    assert not recurrent_state.is_contiguous()
    state_indices = torch.tensor([1, 2, 3] if has_empty_sequence else [1, 3], dtype=torch.int64)
    has_initial_state = torch.tensor([True, True, False] if has_empty_sequence else [True, False])
    metadata = SimpleNamespace(
        cu_seqlens_host=(0, 1, 1, 3) if has_empty_sequence else (0, 1, 3),
        cu_seqlens_kern=(0, 1, 3) if has_empty_sequence else None,
        keep_meta=torch.tensor([True, False, True]) if has_empty_sequence else None,
        chunk_indices_chunk64_host=(0, 0),
    )
    output = torch.randn_like(v)
    final_state = torch.randn(2, 1, 2, 2, dtype=torch.float32)

    def gather(state, indices, flags):
        return state[indices].masked_fill(~flags[:, None, None, None], 0).contiguous()

    # Mock the device-copy boundary; the real plan's kernels are covered by
    # KDA state-copy NPU tests. Keep the padded cache view intact.
    state_copy = SimpleNamespace(
        seal=MagicMock(),
        gather=MagicMock(side_effect=gather),
        scatter=MagicMock(
            side_effect=lambda state, packed, indices: state.__setitem__(indices, packed.to(state.dtype))
        ),
    )
    attention.kv_cache = (None, recurrent_state)
    # Run the actual worker initializer: the class marker must opt A3 into the
    # fallback plan as well as A5 into the fused plan. Only device compilation
    # and copying are mocked; cache binding and publication remain real.
    with (
        patch.object(kda_state_copy_plan, "supports_kda_state_copy", return_value=supports_fused_copy),
        patch.object(kda_state_copy_plan.KDAStateCopyPlan, "prepare", return_value=state_copy) as prepare_fused,
        patch.object(
            kda_state_copy_plan.StridedKDAFallbackPlan, "prepare", return_value=state_copy
        ) as prepare_fallback,
    ):
        kda_state_copy_plan.initialize_kda_state_copy({"ling_kda": attention}, max_selected=8)

    selected_prepare = prepare_fused if supports_fused_copy else prepare_fallback
    unused_prepare = prepare_fallback if supports_fused_copy else prepare_fused
    selected_prepare.assert_called_once_with(recurrent_state, 8)
    unused_prepare.assert_not_called()
    state_copy.seal.assert_called_once()
    assert attention._kda_state_copy_ready is True
    assert attention._ascend_kda_state_copy._layer_name == "ling_kda"
    torch.testing.assert_close(backing, original_backing)
    with (
        patch(
            "vllm_ascend.ops.bailing_moe_v3_kda.run_chunk_kda",
            return_value=(output, final_state),
        ) as run_chunk,
    ):
        actual = attention._run_prefill(
            q,
            k,
            v,
            raw_gate,
            beta,
            recurrent_state,
            state_indices,
            has_initial_state,
            metadata,
        )

    assert actual is output
    initial_state = run_chunk.call_args.args[5]
    assert initial_state.is_contiguous()
    assert initial_state.dtype == cache_dtype
    state_copy.gather.assert_called_once()
    gathered_cache, gathered_indices, gathered_flags = state_copy.gather.call_args.args
    assert gathered_cache is recurrent_state
    torch.testing.assert_close(gathered_indices, torch.tensor([1, 3]))
    torch.testing.assert_close(gathered_flags, torch.tensor([True, False]))
    state_copy.scatter.assert_called_once()
    scattered_cache, scattered_state, scattered_indices = state_copy.scatter.call_args.args
    assert scattered_cache is recurrent_state
    assert scattered_state is final_state
    assert scattered_indices is gathered_indices
    torch.testing.assert_close(initial_state[0], original_state[1])
    assert torch.count_nonzero(initial_state[1]) == 0
    expected_args = (
        q,
        k,
        v,
        raw_gate,
        beta,
        initial_state,
        metadata.cu_seqlens_kern if has_empty_sequence else metadata.cu_seqlens_host,
        metadata.chunk_indices_chunk64_host,
        attention.A_log,
        attention.dt_bias,
    )
    assert len(run_chunk.call_args.args) == len(expected_args)
    for actual_arg, expected_arg in zip(run_chunk.call_args.args, expected_args):
        assert actual_arg is expected_arg
    assert run_chunk.call_args.kwargs == {"lower_bound": expected_lower_bound}
    torch.testing.assert_close(recurrent_state[[1, 3]], final_state.to(cache_dtype))
    torch.testing.assert_close(recurrent_state[[0, 2]], original_state[[0, 2]])
    torch.testing.assert_close(backing[::2], original_backing[::2])


@pytest.mark.parametrize("plan_present, ready", [(False, False), (False, True), (True, False)])
def test_prefill_rejects_unprepared_state_copy_plan(plan_present, ready):
    attention = _new_attention()
    state_copy = SimpleNamespace(gather=MagicMock(), scatter=MagicMock())
    attention._ascend_kda_state_copy = state_copy if plan_present else None
    attention._kda_state_copy_ready = ready
    q = torch.empty(1, 1, 1, 2)
    state = torch.empty(1, 1, 2, 2)
    metadata = SimpleNamespace(cu_seqlens_host=(0, 1), cu_seqlens_kern=None, keep_meta=None)

    with (
        patch("vllm_ascend.ops.bailing_moe_v3_kda.run_chunk_kda") as run_chunk,
        pytest.raises(RuntimeError, match="worker cache initialization before prefill"),
    ):
        attention._run_prefill(q, q, q, q, q, state, torch.tensor([0]), torch.tensor([True]), metadata)

    state_copy.gather.assert_not_called()
    state_copy.scatter.assert_not_called()
    run_chunk.assert_not_called()


def test_prefill_requires_initial_state_metadata():
    attention = _new_attention()
    state_copy = SimpleNamespace(gather=MagicMock(), scatter=MagicMock())
    attention._ascend_kda_state_copy = state_copy
    attention._kda_state_copy_ready = True
    q = torch.empty(1, 1, 1, 2)
    state = torch.empty(1, 1, 2, 2)
    metadata = SimpleNamespace(cu_seqlens_host=(0, 1), cu_seqlens_kern=None, keep_meta=None)

    with (
        patch("vllm_ascend.ops.bailing_moe_v3_kda.run_chunk_kda") as run_chunk,
        pytest.raises(ValueError, match="has_initial_state metadata"),
    ):
        attention._run_prefill(q, q, q, q, q, state, torch.tensor([0]), None, metadata)

    state_copy.gather.assert_not_called()
    state_copy.scatter.assert_not_called()
    run_chunk.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
def test_prefill_reuses_batch_cu_across_layers_without_changing_ascendc(enabled):
    attention = _new_attention()
    attention.safe_gate, attention.lower_bound = True, -5.0
    attention.A_log, attention.dt_bias = torch.zeros(1), torch.zeros(2)
    attention._kda_state_copy_ready = True
    state = torch.zeros(1, 1, 2, 2)
    attention._ascend_kda_state_copy = SimpleNamespace(gather=MagicMock(return_value=state), scatter=MagicMock())
    q = torch.zeros(1, 3, 1, 2)
    prepared = torch.tensor([0, 3], dtype=torch.int64)
    metadata = SimpleNamespace(
        cu_seqlens_host=(0, 3),
        cu_seqlens_kern=None,
        keep_meta=None,
        chunk_indices_chunk64_host=(0, 0),
        flash_cu_seqlens=prepared,
    )
    with (
        patch("vllm_ascend.ops.bailing_moe_v3_kda.FLASH_KDA_ENABLED", enabled),
        patch("vllm_ascend.ops.bailing_moe_v3_kda.run_chunk_kda", return_value=(q, state)) as run_chunk,
    ):
        assert attention.get_attn_backend().get_builder_cls().build_flash_kda_cu_seqlens is enabled
        for _ in range(3):
            attention._run_prefill(q, q, q, q, q, state, torch.tensor([0]), torch.tensor([True]), metadata)
        for call in run_chunk.call_args_list:
            assert call.args[6] is metadata.cu_seqlens_host
            if enabled:
                assert call.kwargs["cu_seqlens_device"] is prepared
            else:
                assert "cu_seqlens_device" not in call.kwargs


def test_flash_prefill_requires_prebuilt_device_cu():
    attention = _new_attention()
    attention.safe_gate, attention.lower_bound = True, -5.0
    state = torch.zeros(1, 1, 2, 2)
    attention._kda_state_copy_ready = True
    attention._ascend_kda_state_copy = SimpleNamespace(gather=MagicMock(return_value=state), scatter=MagicMock())
    q = torch.zeros(1, 3, 1, 2)
    metadata = SimpleNamespace(cu_seqlens_host=(0, 3), cu_seqlens_kern=None, keep_meta=None, flash_cu_seqlens=None)
    with (
        patch("vllm_ascend.ops.bailing_moe_v3_kda.FLASH_KDA_ENABLED", True),
        pytest.raises(RuntimeError, match="batch device cu_seqlens"),
    ):
        attention._run_prefill(q, q, q, q, q, state, torch.tensor([0]), torch.tensor([True]), metadata)


@pytest.mark.parametrize("layout", ["sd", "ds", "page_strided"])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("run_mode", [0, 1])
def test_causal_conv_updates_only_active_cache_rows(layout, cache_dtype, run_mode):
    """Only explicit pad/null sentinels may bypass FLA cache bounds."""
    attention = _new_attention()
    attention._conv_state_dim_first = layout == "ds"
    mixed_qkv = torch.randn(6, 6, dtype=torch.bfloat16)
    conv_weights_t = torch.randn(3, 6, dtype=torch.bfloat16)
    shape = (9, 6, 3) if layout == "ds" else (9, 3, 6)
    backing = torch.arange(18 * 18, dtype=cache_dtype).reshape(18, *shape[1:]) % 13
    storage = backing[::2] if layout == "page_strided" else backing[:9]
    original_backing = backing.clone()
    state_sd = storage.transpose(-1, -2) if layout == "ds" else storage
    # Include a reserved null block, an empty request and a padding index.
    query_start_loc = torch.tensor([0, 2, 3, 3, 4, 6], dtype=torch.int32)
    cache_indices = torch.tensor([[2, 8], [NULL_BLOCK_ID, 8], [5, 8], [PAD_SLOT_ID, 8], [4, 8]], dtype=torch.int32)
    initial_state_mode = torch.tensor([True, False, True, False, True])
    accepted = torch.ones(5, dtype=torch.int32)
    result = torch.full_like(mixed_qkv, 11)
    staged = layout == "ds" or cache_dtype != mixed_qkv.dtype

    def copy_state(cache, packed, indices, starts, packed_indices, *, write_back):
        assert packed.shape == (5, 3, 6)  # Requests, never the nine-slot pool.
        assert packed.dtype == mixed_qkv.dtype
        assert packed.is_contiguous()
        for request, slot in enumerate(indices.tolist()):
            active = 0 <= slot < cache.shape[0] and starts[request + 1] > starts[request]
            if write_back:
                if active:
                    cache[slot].copy_(packed[request])
            else:
                packed[request].copy_(cache[slot] if active else torch.zeros_like(packed[request]))
                packed_indices[request] = request if active else PAD_SLOT_ID

    def apply_conv(input_, weight, state, indices, kwargs):
        assert input_ is mixed_qkv and weight is conv_weights_t
        assert state.dtype == mixed_qkv.dtype
        assert state.is_contiguous() if staged else state.stride(-1) == 1
        assert indices.dim() == 1 and indices.is_contiguous()
        assert kwargs["query_start_loc"] is query_start_loc
        assert kwargs["activation"] == "silu"
        expected_null_block_id = cache_indices.shape[0] if staged else NULL_BLOCK_ID
        assert kwargs["null_block_id"] == expected_null_block_id
        if staged:
            assert indices.tolist() == [0, expected_null_block_id, expected_null_block_id, expected_null_block_id, 4]
        else:
            assert state is storage
            assert indices.tolist() == [2, NULL_BLOCK_ID, NULL_BLOCK_ID, NULL_BLOCK_ID, 4]
        pad_slot_id = kwargs.get("pad_slot_id")
        for request, slot in enumerate(indices.tolist()):
            if slot == kwargs["null_block_id"] or slot == pad_slot_id:
                continue
            assert 0 <= slot < state.shape[0]
            if query_start_loc[request + 1] == query_start_loc[request]:
                continue
            state[slot].add_(7)
        return result

    def prefill(input_, weight, bias, **kwargs):
        assert bias is None
        assert kwargs["has_initial_state"] is initial_state_mode
        assert kwargs["pad_slot_id"] == PAD_SLOT_ID
        return apply_conv(input_, weight, kwargs["conv_states"], kwargs["cache_indices"], kwargs)

    def decode(input_, state, weight, **kwargs):
        assert kwargs["num_accepted_tokens"] is accepted
        assert kwargs["max_query_len"] == 2
        assert torch.count_nonzero(kwargs["out"]) == 0
        return apply_conv(input_, weight, state, kwargs["conv_state_indices"], kwargs)

    with (
        patch("vllm_ascend.ops.bailing_moe_v3_kda.copy_conv_state", side_effect=copy_state) as copy_op,
        patch("vllm_ascend.ops.kda_conv.causal_conv1d_fn", side_effect=prefill) as prefill_op,
        patch("vllm_ascend.ops.kda_conv.causal_conv1d_update", side_effect=decode) as decode_op,
    ):
        actual = attention._run_causal_conv1d(
            mixed_qkv,
            conv_weights_t,
            storage,
            query_start_loc,
            cache_indices,
            initial_state_mode,
            run_mode=run_mode,
            max_query_len=2,
            num_accepted_tokens=accepted if run_mode else None,
        )

    assert actual is result
    assert prefill_op.call_count == (run_mode == 0)
    assert decode_op.call_count == (run_mode == 1)
    assert copy_op.call_count == (2 if staged else 0)
    expected = original_backing
    for slot in (2, 4):
        expected[slot * (2 if layout == "page_strided" else 1)].add_(7)
    torch.testing.assert_close(backing, expected)
    assert state_sd.shape == (9, 3, 6)


@pytest.mark.parametrize("speculative_tokens", [0, 3])
def test_native_single_token_conv_defines_skipped_rows_without_changing_cache_ownership(speculative_tokens):
    attention = _new_attention()
    attention._conv_state_dim_first = False
    attention.num_speculative_tokens = speculative_tokens
    x = torch.randn(6, 6, dtype=torch.bfloat16)
    cache = torch.zeros(8, 3, 6, dtype=torch.bfloat16)
    indices = torch.tensor([2, NULL_BLOCK_ID, 5, PAD_SLOT_ID, 8, 4], dtype=torch.int32)
    starts = torch.tensor([0, 1, 2, 2, 3, 4, 5], dtype=torch.int32)
    valid = torch.tensor([True, False, False, False, False, True])

    def prepare(cache_indices, query_start_loc, num_slots, null_block_id, out):
        assert cache_indices is indices and query_start_loc is starts
        assert num_slots == 8 and null_block_id == NULL_BLOCK_ID
        out.fill_(torch.nan)
        out[~valid] = 0
        return torch.where(valid, indices, null_block_id)

    def native(input_, state, _weight, **kwargs):
        assert state is cache and input_ is x
        assert kwargs["conv_state_indices"].tolist() == [2, 0, 0, 0, 0, 4]
        out = kwargs["out"]
        torch.testing.assert_close(out[~valid], torch.zeros_like(out[~valid]), rtol=0, atol=0)
        if speculative_tokens == 0:
            assert torch.isnan(out[valid]).all()
        else:
            assert torch.count_nonzero(out) == 0
        out[valid] = input_[valid] + 1
        state[[2, 4]] += 1
        return out

    with (
        patch("vllm_ascend.ops.bailing_moe_v3_kda.prepare_native_conv_decode", side_effect=prepare) as fused,
        patch("vllm_ascend.ops.kda_conv.causal_conv1d_update", side_effect=native),
    ):
        out = attention._run_causal_conv1d(
            x,
            torch.empty(4, 6, dtype=x.dtype),
            cache,
            starts,
            indices,
            None,
            run_mode=1,
            max_query_len=1,
        )

    assert fused.call_count == (1 if speculative_tokens == 0 else 0)
    expected = torch.zeros_like(x)
    expected[valid] = x[valid] + 1
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    expected_cache = torch.zeros_like(cache)
    expected_cache[[2, 4]] = 1
    torch.testing.assert_close(cache, expected_cache, rtol=0, atol=0)


def test_causal_conv_packed_all_null_mrv2_uses_nonnegative_sentinel():
    """FLA update filters only a configured nonnegative null-block ID."""
    attention = _new_attention()
    attention._conv_state_dim_first = False
    requests, tokens_per_request = 32, 4
    state_len, dim = 6, 6
    mixed_qkv = torch.randn(requests * tokens_per_request, dim, dtype=torch.bfloat16)
    conv_weights_t = torch.randn(4, dim, dtype=torch.bfloat16)
    conv_state = torch.randn(8, state_len, dim, dtype=torch.float32)
    original_state = conv_state.clone()
    query_start_loc = torch.arange(
        0,
        requests * tokens_per_request + 1,
        tokens_per_request,
        dtype=torch.int32,
    )
    cache_indices = torch.full((requests, tokens_per_request), NULL_BLOCK_ID, dtype=torch.int32)
    accepted = torch.ones(requests, dtype=torch.int32)

    def copy_state(cache, packed, indices, starts, packed_indices, *, write_back):
        assert cache is conv_state
        assert packed.shape == (requests, state_len, dim)
        assert packed.dtype == mixed_qkv.dtype
        assert torch.equal(indices, torch.full_like(indices, PAD_SLOT_ID))
        assert starts is query_start_loc
        if not write_back:
            packed.zero_()
            packed_indices.fill_(PAD_SLOT_ID)

    def decode(input_, state, weight, **kwargs):
        assert input_ is mixed_qkv and weight is conv_weights_t
        assert state.shape == (requests, state_len, dim)
        assert state.dtype == mixed_qkv.dtype
        assert kwargs["query_start_loc"] is query_start_loc
        assert kwargs["num_accepted_tokens"] is accepted
        assert kwargs["max_query_len"] == tokens_per_request
        assert kwargs["null_block_id"] == requests
        indices = kwargs["conv_state_indices"]
        assert indices.tolist() == [requests] * requests
        assert torch.all(indices >= 0)
        assert torch.count_nonzero(kwargs["out"]) == 0
        return kwargs["out"]

    with (
        patch("vllm_ascend.ops.bailing_moe_v3_kda.copy_conv_state", side_effect=copy_state) as copy_op,
        patch("vllm_ascend.ops.kda_conv.causal_conv1d_update", side_effect=decode) as decode_op,
    ):
        actual = attention._run_causal_conv1d(
            mixed_qkv,
            conv_weights_t,
            conv_state,
            query_start_loc,
            cache_indices,
            None,
            run_mode=1,
            max_query_len=tokens_per_request,
            num_accepted_tokens=accepted,
        )

    assert torch.equal(actual, torch.zeros_like(actual))
    torch.testing.assert_close(conv_state, original_state)
    assert copy_op.call_count == 2
    decode_op.assert_called_once()


@pytest.mark.parametrize("metadata", [None, {}, {"other_layer": object()}])
def test_profile_clears_preallocated_output(metadata):
    attention = _new_attention()
    attention.prefix = "model.layers.0.linear_attn"
    core_attn_out = torch.full((1, 5, 1, 2), torch.nan)

    with patch(
        "vllm_ascend.ops.bailing_moe_v3_kda.get_forward_context",
        return_value=SimpleNamespace(attn_metadata=metadata),
    ):
        attention._forward(
            torch.empty(5, 2),
            torch.empty(5, 2),
            torch.empty(5, 2),
            torch.empty(1, 5, 1, 2),
            torch.empty(1, 5, 1),
            core_attn_out,
        )

    assert torch.equal(core_attn_out, torch.zeros_like(core_attn_out))


def test_idle_metadata_clears_entire_uninitialized_output(monkeypatch):
    _patch_missing_runtime_cdiv(monkeypatch)
    _, _, metadata = _build_attn_metadata(
        BatchSpec(seq_lens=[10], query_lens=[1]),
        num_speculative_tokens=0,
        num_decode_draft_tokens_cpu=None,
    )
    metadata.num_actual_tokens = 0
    metadata.num_prefills = 0
    metadata.num_decodes = 0
    attention = SimpleNamespace(
        prefix="model.layers.0.linear_attn",
        head_dim=2,
        kv_cache=(torch.empty(1), torch.empty(1)),
        _conv_weights_t=MagicMock(return_value=torch.empty(1)),
        _run_causal_conv1d=MagicMock(),
        _run_recurrent=MagicMock(),
        _run_prefill=MagicMock(),
    )
    core_attn_out = torch.full((1, 3, 1, 2), torch.nan)

    with patch(
        "vllm_ascend.ops.bailing_moe_v3_kda.get_forward_context",
        return_value=SimpleNamespace(attn_metadata={attention.prefix: metadata}),
    ):
        AscendBailingMoeV3KimiDeltaAttention._forward(
            attention,
            torch.empty(3, 2),
            torch.empty(3, 2),
            torch.empty(3, 2),
            torch.empty(1, 3, 1, 2),
            torch.empty(1, 3, 1),
            core_attn_out,
        )

    assert torch.equal(core_attn_out, torch.zeros_like(core_attn_out))
    attention._run_causal_conv1d.assert_not_called()
    attention._run_recurrent.assert_not_called()
    attention._run_prefill.assert_not_called()


def test_spec_padding_discards_unwritten_nan_output():
    output = torch.full((1, 6, 1, 2), torch.nan)
    output[:, :4] = 3
    query_start_loc = torch.tensor([0, 2, 4, 4], dtype=torch.int32)
    actual = _zero_padded_spec_output(output, query_start_loc)
    torch.testing.assert_close(actual[:, :4], output[:, :4])
    assert torch.equal(actual[:, 4:], torch.zeros_like(actual[:, 4:]))


@pytest.mark.parametrize(
    ("seq_lens", "query_lens", "draft_tokens"),
    [
        pytest.param([3, 4], [3, 2], None, id="prefill"),
        pytest.param([10, 20], [1, 1], None, id="decode"),
        pytest.param([10, 8, 7], [1, 4, 2], None, id="decode-prefill"),
        pytest.param([10, 20], [3, 3], [2, 2], id="spec"),
        pytest.param([10, 7, 20], [3, 4, 1], [2, -1, -1], id="spec-prefill-decode"),
    ],
)
@pytest.mark.parametrize("flash_kda_enabled", [False, True])
def test_forward_routes_real_builder_metadata_and_clears_tail(
    monkeypatch, seq_lens, query_lens, draft_tokens, flash_kda_enabled, extra_unassigned_tokens=0
):
    _patch_missing_runtime_cdiv(monkeypatch)
    num_speculative_tokens = 2 if draft_tokens is not None else 0
    _, _, metadata = _build_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens),
        num_speculative_tokens=num_speculative_tokens,
        num_decode_draft_tokens_cpu=None if draft_tokens is None else torch.tensor(draft_tokens, dtype=torch.int32),
    )
    num_actual_tokens = sum(query_lens) + extra_unassigned_tokens
    metadata.num_actual_tokens = num_actual_tokens
    num_allocated_tokens = num_actual_tokens + 2
    head_dim = 2
    q = torch.arange(num_allocated_tokens * head_dim, dtype=torch.float32).reshape(-1, head_dim)
    k = q + 100
    v = q + 200
    raw_gate = torch.zeros(1, num_allocated_tokens, 1, head_dim)
    beta_logits = torch.linspace(-8, 8, num_allocated_tokens, dtype=torch.bfloat16).reshape(1, -1, 1)
    beta = beta_logits if flash_kda_enabled else beta_logits.float().sigmoid()

    def conv_forward(mixed, _weight, _state, starts, *_args, **_kwargs):
        assert starts[-1] == mixed.shape[0]
        return mixed

    def recurrent_forward(_q, _k, value, _gate, _beta, _state, starts, *_args, **_kwargs):
        assert starts[-1] == value.shape[1]
        return value + 10

    def prefill_forward(_q, _k, value, _gate, _beta, _state, state_indices, has_state, chunk_metadata):
        assert chunk_metadata.cu_seqlens_host[-1] == value.shape[1]
        assert state_indices is metadata.prefill_state_indices
        assert has_state is metadata.prefill_has_initial_state
        return value + 20

    conv = MagicMock(side_effect=conv_forward)
    recurrent = MagicMock(side_effect=recurrent_forward)
    prefill = MagicMock(side_effect=prefill_forward)
    attention = SimpleNamespace(
        prefix="model.layers.0.linear_attn",
        head_dim=head_dim,
        num_speculative_tokens=num_speculative_tokens,
        kv_cache=(torch.zeros(1), torch.zeros(1)),
        _conv_weights_t=lambda _dtype: torch.empty(1),
        _run_causal_conv1d=conv,
        _run_recurrent=recurrent,
        _run_prefill=prefill,
    )
    core_attn_out = torch.full((1, num_allocated_tokens, 1, head_dim), torch.nan)

    with (
        patch(
            "vllm_ascend.ops.bailing_moe_v3_kda.get_forward_context",
            return_value=SimpleNamespace(attn_metadata={attention.prefix: metadata}),
        ),
        patch("vllm_ascend.ops.bailing_moe_v3_kda.FLASH_KDA_ENABLED", flash_kda_enabled),
    ):
        AscendBailingMoeV3KimiDeltaAttention._forward(
            attention,
            q,
            k,
            v,
            raw_gate,
            beta,
            core_attn_out,
        )

    expected = torch.zeros_like(core_attn_out)
    if metadata.spec_sequence_masks is not None:
        assert metadata.num_decodes == 0
        spec_indices = metadata.spec_token_indx
        expected[0, spec_indices, 0] = v[spec_indices] + 10
        if metadata.num_prefills:
            non_spec_indices = metadata.non_spec_token_indx
            expected[0, non_spec_indices, 0] = v[non_spec_indices] + 20
        assert (
            recurrent.call_args.kwargs["num_accepted_tokens"]
            is metadata.spec_decode_metadata.spec_causal_conv1d.num_accepted_tokens
        )
        assert conv.call_args_list[0].kwargs["max_query_len"] == num_speculative_tokens + 1
    else:
        split = metadata.num_decode_tokens
        expected[0, :split, 0] = v[:split] + 10
        expected[0, split:num_actual_tokens, 0] = v[split:num_actual_tokens] + 20
    assert prefill.call_count == (metadata.num_prefills > 0)
    assert recurrent.call_count == (metadata.num_decodes > 0 or metadata.spec_sequence_masks is not None)
    assert conv.call_count == (2 if metadata.num_prefills and metadata.spec_sequence_masks is not None else 1)

    expected_beta = beta[:, :num_actual_tokens]
    if recurrent.called:
        if metadata.spec_sequence_masks is not None:
            recurrent_beta = (
                expected_beta
                if metadata.num_prefills == 0 and metadata.num_decodes == 0
                else expected_beta.index_select(1, metadata.spec_token_indx)
            )
        else:
            recurrent_beta = expected_beta[:, : metadata.num_decode_tokens]
        torch.testing.assert_close(recurrent.call_args.args[4], recurrent_beta, rtol=0, atol=0)
    if prefill.called:
        prefill_beta = (
            expected_beta.index_select(1, metadata.non_spec_token_indx)
            if metadata.spec_sequence_masks is not None
            else expected_beta[:, metadata.num_decode_tokens :]
        )
        torch.testing.assert_close(prefill.call_args.args[4], prefill_beta, rtol=0, atol=0)

    torch.testing.assert_close(core_attn_out, expected)
    torch.testing.assert_close(
        core_attn_out[:, num_actual_tokens:],
        torch.zeros_like(core_attn_out[:, num_actual_tokens:]),
    )


@pytest.mark.parametrize("flash_kda_enabled", [False, True])
def test_mixed_spec_scatter_clears_unassigned_graph_rows(monkeypatch, flash_kda_enabled):
    # Neither index_copy writes the extra actual-token row. It must be zeroed
    # before scattering, even though complete single-path copies need no fill.
    test_forward_routes_real_builder_metadata_and_clears_tail(
        monkeypatch,
        seq_lens=[10, 7, 20],
        query_lens=[3, 4, 1],
        draft_tokens=[2, -1, -1],
        flash_kda_enabled=flash_kda_enabled,
        extra_unassigned_tokens=1,
    )


@pytest.mark.parametrize("num_tokens", [1, 2, 4])
@pytest.mark.parametrize("width,gap", [(3, 7), (512, 4), (512, 1040)])
def test_decode_qkv_compaction_can_defer_generic_adjacent_row_gaps(num_tokens, width, gap):
    row_stride, offset = 3 * width + gap, 16
    backing = torch.arange(offset + num_tokens * row_stride, dtype=torch.float32).to(torch.bfloat16)
    projection = backing.as_strided((num_tokens, 3 * width), (row_stride, 1), offset)
    q, k, v = projection.chunk(3, dim=-1)
    expected = torch.cat((q, k, v), dim=-1)
    original = backing.clone()
    with patch("vllm_ascend.ops.bailing_moe_v3_kda.torch.cat", side_effect=AssertionError("defer safe row-gap cat")):
        actual = _contiguous_packed_qkv(q, k, v, allow_row_gaps=True)
    assert actual.stride() == projection.stride()
    assert actual.storage_offset() == offset
    assert actual.untyped_storage().data_ptr() == backing.untyped_storage().data_ptr()
    assert actual.is_contiguous() == (num_tokens == 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(backing, original, rtol=0, atol=0)


@pytest.mark.parametrize(
    "invalid", ["state_dtype", "input_dtype", "index_dtype", "starts_dtype", "rows", "starts_shape", "speculative"]
)
def test_decode_qkv_defer_guard_rejects_non_native_or_non_fixed_decode(invalid):
    rows, input_dtype = 2, torch.bfloat16
    state = torch.empty(8, 3, 12, dtype=torch.bfloat16)
    indices, starts, speculative = torch.arange(rows, dtype=torch.int32), torch.arange(rows + 1, dtype=torch.int32), 0
    assert _native_decode_prepare_compatible(rows, input_dtype, state, indices, starts, speculative)
    if invalid == "state_dtype":
        state = state.float()
    elif invalid == "input_dtype":
        input_dtype = torch.float16
    elif invalid == "index_dtype":
        indices = indices.long()
    elif invalid == "starts_dtype":
        starts = starts.long()
    elif invalid == "rows":
        rows += 1
    elif invalid == "starts_shape":
        starts = starts[:-1]
    else:
        speculative = 3
    assert not _native_decode_prepare_compatible(rows, input_dtype, state, indices, starts, speculative)


@pytest.mark.parametrize("num_tokens,gap", [(1, 4), (2, 4), (4, 1040)])
def test_native_single_token_decode_folds_row_gap_pack_into_prepare(num_tokens, gap):
    attention = _new_attention()
    attention._conv_state_dim_first = False
    attention.num_speculative_tokens = 0
    dim = 12
    storage = torch.randn(num_tokens, dim + gap, dtype=torch.bfloat16)
    x = storage[:, :dim]
    expected = x.clone()
    state = torch.empty(8, 3, dim, dtype=x.dtype)
    indices = torch.arange(1, num_tokens + 1, dtype=torch.int32)
    starts = torch.arange(num_tokens + 1, dtype=torch.int32)

    def prepare(cache_indices, query_start_loc, slots, null, out, **kwargs):
        assert slots == state.shape[0] and null == NULL_BLOCK_ID
        assert cache_indices is indices and query_start_loc is starts
        if num_tokens == 1:
            assert kwargs == {}
        else:
            assert kwargs["input_tensor"] is x
            assert kwargs["packed_input"].is_contiguous()
            kwargs["packed_input"].copy_(x)
        return indices

    def native(input_, cache, _weight, **kwargs):
        assert cache is state
        assert input_.is_contiguous()
        assert (input_ is x) == (num_tokens == 1)
        torch.testing.assert_close(input_, expected, rtol=0, atol=0)
        out = kwargs["out"]
        assert out.is_contiguous()
        out.copy_(input_)
        return out

    with (
        patch("vllm_ascend.ops.bailing_moe_v3_kda.prepare_native_conv_decode", side_effect=prepare) as prepared,
        patch("vllm_ascend.ops.kda_conv.causal_conv1d_update", side_effect=native) as update,
    ):
        actual = attention._run_causal_conv1d(
            x, torch.empty(4, dim, dtype=x.dtype), state, starts, indices, None, run_mode=1, max_query_len=1
        )
    assert prepared.call_count == update.call_count == 1
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class _InputProjectionLinear(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.quant_method = UnquantizedLinearMethod()
        self.bias = None
        self.gather_output = False
        self.custom_op = None

    def forward(self, hidden):
        return torch.nn.functional.linear(hidden, self.weight), None


def _input_projection_attention(
    *,
    enabled=None,
    num_speculative_tokens=0,
    separate_b_proj=False,
    dtype=torch.bfloat16,
):
    projection, heads, hidden = 16, 2, 8

    def fake_base_init(instance, *_args, **_kwargs):
        nn.Module.__init__(instance)
        instance.num_speculative_tokens = num_speculative_tokens
        instance.separate_b_proj = separate_b_proj
        instance.hidden_size = hidden
        instance.head_dim = projection // heads
        instance.local_num_heads = heads
        instance.projection_size_per_partition = projection
        instance.model_config = SimpleNamespace(dtype=dtype)
        instance.prefix = "model.layers.0.linear_attn"
        instance.conv_size = 4
        instance.q_conv1d, instance.k_conv1d, instance.v_conv1d = (
            nn.Module(),
            nn.Module(),
            nn.Module(),
        )
        for conv in (instance.q_conv1d, instance.k_conv1d, instance.v_conv1d):
            conv.weight = nn.Parameter(torch.ones(projection, 1, 4), requires_grad=False)
        qkvb = torch.cat(
            [torch.full((projection, hidden), value, dtype=dtype) for value in (1.0, 2.0, 3.0)]
            + [torch.full((heads, hidden), 4.0, dtype=dtype)],
            dim=0,
        )
        instance.qkvb_proj = _InputProjectionLinear(qkvb)
        instance.f_proj = _InputProjectionLinear(torch.full((projection, hidden), 5.0, dtype=dtype))
        instance.g_proj = _InputProjectionLinear(torch.full((projection, hidden), 6.0, dtype=dtype))
        instance.o_proj = _InputProjectionLinear(torch.ones(hidden, projection, dtype=dtype))
        instance.o_norm = _RecordingNorm()

    config = SimpleNamespace() if enabled is None else SimpleNamespace(fuse_kda_input_projections=enabled)
    with (
        patch.object(BailingMoeV3KimiDeltaAttention, "__init__", fake_base_init),
        patch(
            "vllm_ascend.ops.bailing_moe_v3_kda.is_conv_state_dim_first",
            return_value=False,
        ),
    ):
        return AscendBailingMoeV3KimiDeltaAttention(config)


def test_input_projection_fusion_default_keeps_canonical_parameters_without_new_buffer():
    attention = _input_projection_attention()
    assert attention._ascend_fuse_input_projections is False
    assert "ascend_input_projection_weight" not in attention._buffers
    assert "qkvb_proj.weight" in dict(attention.named_parameters())
    assert "f_proj.weight" in dict(attention.named_parameters())
    assert "g_proj.weight" in dict(attention.named_parameters())


@pytest.mark.parametrize("unsupported", ["mtp", "separate_beta", "fp32", "non_boolean"])
def test_input_projection_fusion_rejects_unsupported_checkpoint_contracts(unsupported):
    args = dict(enabled=True)
    if unsupported == "mtp":
        args["num_speculative_tokens"] = 1
    elif unsupported == "separate_beta":
        args["separate_b_proj"] = True
    elif unsupported == "fp32":
        args["dtype"] = torch.float32
    else:
        args["enabled"] = "true"
    with pytest.raises(ValueError, match="KDA|fuse_kda_input_projections"):
        _input_projection_attention(**args)


@pytest.mark.parametrize("unsupported", ["quantized", "bias", "gather", "custom_parallel_op", "model_dtype"])
def test_input_projection_fusion_does_not_bypass_linear_semantics(unsupported):
    attention = _input_projection_attention(enabled=True)
    if unsupported == "quantized":
        attention.f_proj.quant_method = object()
    elif unsupported == "bias":
        attention.f_proj.bias = nn.Parameter(torch.zeros(16, dtype=torch.bfloat16), requires_grad=False)
    elif unsupported == "gather":
        attention.f_proj.gather_output = True
    elif unsupported == "custom_parallel_op":
        attention.f_proj.custom_op = object()
    else:
        attention.model_config.dtype = torch.float32
    with pytest.raises(ValueError, match="BF16"):
        attention._pack_input_projection_weights()


def test_input_projection_pack_order_keeps_beta_after_aligned_qkv_and_gates():
    attention = _input_projection_attention(enabled=True)
    assert attention.get_buffer("ascend_input_projection_weight") is None
    attention._pack_input_projection_weights()
    weight = attention.get_buffer("ascend_input_projection_weight")
    # Tagged canonical rows Q/K/V/B/F/G = 1/2/3/4/5/6 must become Q/K/V/F/G/B.
    sections = weight.split((16, 16, 16, 16, 16, 2, 14), dim=0)
    for section, expected in zip(sections, (1, 2, 3, 5, 6, 4, 0)):
        torch.testing.assert_close(section, torch.full_like(section, expected), rtol=0, atol=0)
    assert weight.shape == (96, 8)
    assert "ascend_input_projection_weight" in attention._non_persistent_buffers_set
    assert "ascend_input_projection_weight" not in attention.state_dict()
    assert "ascend_input_projection_weight" not in dict(attention.named_parameters())


def test_input_projection_refit_refreshes_values_without_replacing_captured_storage():
    attention = _input_projection_attention(enabled=True)
    attention._pack_input_projection_weights()
    weight = attention.get_buffer("ascend_input_projection_weight")
    pointer = weight.data_ptr()
    canonical_parameters = {name: id(parameter) for name, parameter in attention.named_parameters()}
    attention.f_proj.weight.fill_(7)
    attention.g_proj.weight.fill_(8)
    attention.qkvb_proj.weight[-2:].fill_(9)
    attention._pack_input_projection_weights()
    assert attention.get_buffer("ascend_input_projection_weight") is weight
    assert weight.data_ptr() == pointer
    assert {name: id(parameter) for name, parameter in attention.named_parameters()} == canonical_parameters
    for section, expected in zip(weight.split((48, 16, 16, 2, 14), dim=0)[1:], (7, 8, 9, 0)):
        torch.testing.assert_close(section, torch.full_like(section, expected), rtol=0, atol=0)
    assert attention._ascend_input_projection_ready is True


@pytest.mark.parametrize("changed", ["shape", "dtype", "device", "format"])
def test_input_projection_refit_rejects_runtime_buffer_contract_changes(changed):
    attention = _input_projection_attention(enabled=True)
    with patch(
        "vllm_ascend.ops.bailing_moe_v3_kda._input_projection_weight_format",
        return_value=2,
    ):
        attention._pack_input_projection_weights()
    weight = attention.get_buffer("ascend_input_projection_weight")
    replacement = weight
    if changed == "shape":
        replacement = torch.empty((97, 8), dtype=weight.dtype)
    elif changed == "dtype":
        replacement = weight.float()
    elif changed == "device":
        replacement = torch.empty(weight.shape, device="meta", dtype=weight.dtype)
    attention.register_buffer("ascend_input_projection_weight", replacement, persistent=False)
    with (
        patch(
            "vllm_ascend.ops.bailing_moe_v3_kda._input_projection_weight_format",
            return_value=29 if changed == "format" else 2,
        ),
        pytest.raises(RuntimeError, match="cannot change after initialization"),
    ):
        attention._pack_input_projection_weights()
    assert attention._ascend_input_projection_ready is False


@pytest.mark.parametrize("tokens", [0, 1, 4, 5, 8, 128, 8192])
@pytest.mark.parametrize("enabled", [False, True])
def test_input_projection_forward_uses_the_configured_path_for_all_token_rows(tokens, enabled):
    attention = _input_projection_attention(enabled=enabled)
    if enabled:
        attention._pack_input_projection_weights()
    hidden = torch.full((tokens, 8), 0.125, dtype=torch.bfloat16)
    output = torch.empty_like(hidden)
    calls = []
    hooks = [
        layer.register_forward_hook(lambda _module, _args, _output, name=name: calls.append(name))
        for name, layer in (
            ("qkvb", attention.qkvb_proj),
            ("f", attention.f_proj),
            ("g", attention.g_proj),
        )
    ]

    def attention_call(q, k, v, gate, beta, core, _prefix):
        core.zero_()
        for value, expected in zip(
            (q, k, v, gate, beta),
            (1.0, 2.0, 3.0, 5.0, torch.sigmoid(torch.tensor(4.0)).item()),
        ):
            torch.testing.assert_close(value, torch.full_like(value, expected), rtol=0, atol=0)

    try:
        with (
            patch("vllm_ascend.ops.bailing_moe_v3_kda.FLASH_KDA_ENABLED", False),
            patch(
                "vllm_ascend.ops.bailing_moe_v3_kda.torch.ops.vllm.unquantized_gemm",
                side_effect=torch.nn.functional.linear,
                create=True,
            ) as gemm,
            patch(
                "vllm_ascend.ops.bailing_moe_v3_kda.torch.ops.vllm.bailing_v3_kda_attention",
                side_effect=attention_call,
            ),
        ):
            attention.forward(hidden, torch.empty(tokens), output)
        output_gate = attention.o_norm.inputs[1]
        torch.testing.assert_close(output_gate, torch.full_like(output_gate, 6.0), rtol=0, atol=0)
        if enabled:
            gemm.assert_called_once()
            assert calls == []
        else:
            gemm.assert_not_called()
            assert calls == ["qkvb", "f", "g"]
    finally:
        for hook in hooks:
            hook.remove()


def test_input_projection_requires_refit_postload_before_small_batch_forward():
    attention = _input_projection_attention(enabled=True)
    with pytest.raises(RuntimeError, match="deferred post-load/refit"):
        attention._run_fused_input_projection(torch.ones(1, 8, dtype=torch.bfloat16))


@pytest.mark.parametrize("backend", ["hccl", "npu_ipc", "nccl", "sparse_nccl"])
def test_input_projection_fusion_requires_postload_refreshing_weight_updates(backend):
    config = SimpleNamespace(weight_transfer_config=SimpleNamespace(backend=backend))
    with patch(
        "vllm_ascend.ops.bailing_moe_v3_kda.get_current_vllm_config_or_none",
        return_value=config,
    ):
        if backend == "sparse_nccl":
            with pytest.raises(ValueError, match="sparse_nccl"):
                _input_projection_attention(enabled=True)
            # The opt-in restriction must not affect the default model path.
            assert _input_projection_attention()._ascend_fuse_input_projections is False
        else:
            _input_projection_attention(enabled=True)


def test_input_projection_real_deferred_finalize_restores_and_refreshes_captured_buffers():
    attention = _input_projection_attention(enabled=True)
    attention.process_weights_after_loading(torch.bfloat16)
    captured = get_layer_params_buffers(attention)
    conv = attention.get_buffer("ascend_conv1d_weight")
    projection = attention.get_buffer("ascend_input_projection_weight")
    original_pointers = (conv.data_ptr(), projection.data_ptr())
    canonical_ids = {name: id(parameter) for name, parameter in attention.named_parameters()}
    for order, values in (("fg", (7, 8, 9)), ("gf", (10, 11, 12))):
        f_value, g_value, b_value = values
        for name in order:
            getattr(attention, name + "_proj").weight.fill_(f_value if name == "f" else g_value)
        attention.qkvb_proj.weight[-2:].fill_(b_value)
        # RL restore may delete the derived buffers, but finalize must restore
        # their captured storage before this parent's post-load repacks them.
        delattr(attention, "ascend_input_projection_weight")
        delattr(attention, "ascend_conv1d_weight")
        info = LayerReloadingInfo(
            restore_metadata=({}, {}),
            restore_device=torch.device("cpu"),
            kernel_tensors=captured,
            kernel_non_persistent_buffers={"ascend_conv1d_weight", "ascend_input_projection_weight"},
        )
        _finalize_attention_layer(attention, info, SimpleNamespace(dtype=torch.bfloat16))
        assert attention.get_buffer("ascend_conv1d_weight") is conv
        assert attention.get_buffer("ascend_input_projection_weight") is projection
        assert (conv.data_ptr(), projection.data_ptr()) == original_pointers
        assert {name: id(parameter) for name, parameter in attention.named_parameters()} == canonical_ids
        for section, expected in zip(projection.split((48, 16, 16, 2, 14), dim=0)[1:], (*values, 0)):
            torch.testing.assert_close(section, torch.full_like(section, expected), rtol=0, atol=0)
        assert attention._ascend_input_projection_ready
        assert attention._ascend_conv_weights_ready
        assert {"ascend_conv1d_weight", "ascend_input_projection_weight"} <= attention._non_persistent_buffers_set


@pytest.mark.parametrize("num_tokens", [1, 2, 4])
def test_fused_projection_uses_j_native_prepare_without_qkv_cat_or_gate_mutation(num_tokens):
    attention = _input_projection_attention(enabled=True)
    attention._pack_input_projection_weights()
    hidden = torch.full((num_tokens, attention.hidden_size), 0.125, dtype=torch.bfloat16)
    with patch(
        "vllm_ascend.ops.bailing_moe_v3_kda.torch.ops.vllm.unquantized_gemm",
        side_effect=torch.nn.functional.linear,
        create=True,
    ):
        qkv, f_values, g_values, beta = attention._run_fused_input_projection(hidden)
    q, k, v = qkv.chunk(3, dim=-1)
    with patch("vllm_ascend.ops.bailing_moe_v3_kda.torch.cat", side_effect=AssertionError("unexpected QKV cat")):
        mixed = _contiguous_packed_qkv(q, k, v, allow_row_gaps=True)
    assert mixed.data_ptr() == qkv.data_ptr()
    assert mixed.stride() == qkv.stride()
    assert mixed.is_contiguous() == (num_tokens == 1)
    before = [value.clone() for value in (qkv, f_values, g_values, beta)]
    channels = qkv.shape[1]
    state = torch.empty(num_tokens + 2, 3, channels, dtype=torch.bfloat16)
    indices = torch.arange(1, num_tokens + 1, dtype=torch.int32)
    starts = torch.arange(num_tokens + 1, dtype=torch.int32)

    def prepare(_indices, _starts, _slots, _null, _output, **kwargs):
        if num_tokens == 1:
            assert kwargs == {}
        else:
            assert kwargs["input_tensor"] is mixed
            assert kwargs["packed_input"].is_contiguous()
            kwargs["packed_input"].copy_(mixed)
        return indices

    def native(value, _state, _weight, **kwargs):
        assert value.is_contiguous()
        torch.testing.assert_close(value, before[0], rtol=0, atol=0)
        kwargs["out"].copy_(value)
        return kwargs["out"]

    with (
        patch("vllm_ascend.ops.bailing_moe_v3_kda.prepare_native_conv_decode", side_effect=prepare) as prepared,
        patch("vllm_ascend.ops.kda_conv.causal_conv1d_update", side_effect=native) as update,
    ):
        actual = attention._run_causal_conv1d(
            mixed,
            torch.empty(4, channels, dtype=torch.bfloat16),
            state,
            starts,
            indices,
            None,
            run_mode=1,
            max_query_len=1,
        )
    assert prepared.call_count == update.call_count == 1
    torch.testing.assert_close(actual, before[0], rtol=0, atol=0)
    for value, expected in zip((qkv, f_values, g_values, beta), before):
        torch.testing.assert_close(value, expected, rtol=0, atol=0)


def test_input_projection_fusion_rejects_lora_without_changing_the_default_path():
    config = SimpleNamespace(lora_config=SimpleNamespace(max_lora_rank=16))
    with patch(
        "vllm_ascend.ops.bailing_moe_v3_kda.get_current_vllm_config_or_none",
        return_value=config,
    ):
        with pytest.raises(ValueError, match="LoRA"):
            _input_projection_attention(enabled=True)
        attention = _input_projection_attention()
        assert attention._ascend_fuse_input_projections is False
        assert "ascend_input_projection_weight" not in attention._buffers


def _raw_gate_native_decode_case():
    attention = _new_attention()
    attention._conv_state_dim_first = False
    attention.num_speculative_tokens = 0
    rows, width, stride = 2, 4, 28
    storage = torch.arange(rows * stride, dtype=torch.float32).bfloat16()
    qkv = storage.as_strided((rows, 3 * width), (stride, 1))
    gate = storage.as_strided((1, rows, 2, 2), (rows * stride, stride, 2, 1), 3 * width)
    values = dict(
        mixed_qkv=qkv,
        conv_weights_t=torch.empty(4, 3 * width, dtype=torch.bfloat16),
        conv_state_storage=torch.empty(8, 3, 3 * width, dtype=torch.bfloat16),
        query_start_loc=torch.arange(rows + 1, dtype=torch.int32),
        cache_indices=torch.arange(1, rows + 1, dtype=torch.int32),
        initial_state_mode=None,
        run_mode=1,
        max_query_len=1,
        gate_tensor=gate,
        packed_gate=torch.empty_like(gate, memory_format=torch.contiguous_format),
    )
    return attention, values


def test_native_decode_gate_pack_uses_zero_copy_2d_metadata_views_and_dense_gate():
    attention, values = _raw_gate_native_decode_case()
    source_gate, target_gate = values["gate_tensor"], values["packed_gate"]
    source_before = values["mixed_qkv"].untyped_storage().clone()

    def prepare(indices, starts, slots, null, output, **options):
        assert slots == 8 and null == NULL_BLOCK_ID
        assert indices is values["cache_indices"] and starts is values["query_start_loc"]
        gate, packed = options["gate_tensor"], options["packed_gate"]
        assert gate.shape == packed.shape == (2, 4)
        assert gate.data_ptr() == source_gate.data_ptr()
        assert gate.storage_offset() == source_gate.storage_offset()
        assert gate.stride() == (28, 1)
        assert packed.data_ptr() == target_gate.data_ptr() and packed.is_contiguous()
        assert gate.untyped_storage().data_ptr() == options["input_tensor"].untyped_storage().data_ptr()
        options["packed_input"].copy_(options["input_tensor"])
        packed.copy_(gate)
        return indices

    def native(input_, state, _weights, **options):
        assert input_.is_contiguous() and state is values["conv_state_storage"]
        torch.testing.assert_close(target_gate, source_gate, rtol=0, atol=0)
        options["out"].copy_(input_)
        return options["out"]

    with (
        patch("vllm_ascend.ops.bailing_moe_v3_kda.prepare_native_conv_decode", side_effect=prepare) as prepared,
        patch("vllm_ascend.ops.kda_conv.causal_conv1d_update", side_effect=native) as update,
    ):
        actual = attention._run_causal_conv1d(**values)
    assert prepared.call_count == update.call_count == 1
    torch.testing.assert_close(actual, values["mixed_qkv"], rtol=0, atol=0)
    assert list(values["mixed_qkv"].untyped_storage()) == list(source_before)


@pytest.mark.parametrize(
    "invalid",
    [
        "state_dtype",
        "index_dtype",
        "starts_dtype",
        "run_mode",
        "max_query",
        "accepted",
        "spec",
        "contiguous",
        "gate_shape",
        "state_alias",
        "indices_alias",
        "starts_alias",
    ],
)
def test_native_decode_gate_pack_rejects_unsupported_cases_before_state_or_native_updates(invalid):
    attention, values = _raw_gate_native_decode_case()
    if invalid == "state_dtype":
        values["conv_state_storage"] = values["conv_state_storage"].float()
    elif invalid == "index_dtype":
        values["cache_indices"] = values["cache_indices"].long()
    elif invalid == "starts_dtype":
        values["query_start_loc"] = values["query_start_loc"].long()
    elif invalid == "run_mode":
        values["run_mode"] = 0
    elif invalid == "max_query":
        values["max_query_len"] = 2
    elif invalid == "accepted":
        values["num_accepted_tokens"] = torch.ones(2, dtype=torch.int32)
    elif invalid == "spec":
        attention.num_speculative_tokens = 1
    elif invalid == "contiguous":
        values["mixed_qkv"] = values["mixed_qkv"].contiguous()
    elif invalid == "gate_shape":
        values["gate_tensor"] = values["gate_tensor"][0]
    elif invalid == "state_alias":
        values["packed_gate"] = values["conv_state_storage"].view(-1)[:8].view(1, 2, 2, 2)
    else:
        key = "cache_indices" if invalid == "indices_alias" else "query_start_loc"
        backing = torch.empty(16, dtype=torch.int32)
        original = values[key]
        values[key] = backing[: original.numel()]
        values[key].copy_(original)
        values["packed_gate"] = backing.view(torch.bfloat16)[:8].view(1, 2, 2, 2)
    state_before = values["conv_state_storage"].clone()
    with (
        patch("vllm_ascend.ops.bailing_moe_v3_kda.prepare_native_conv_decode") as prepared,
        patch("vllm_ascend.ops.bailing_moe_v3_kda.copy_conv_state") as copy,
        patch("vllm_ascend.ops.bailing_moe_v3_kda.gather_update_conv_state") as gather,
        patch("vllm_ascend.ops.kda_conv.causal_conv1d_update") as native,
        pytest.raises(ValueError),
    ):
        attention._run_causal_conv1d(**values)
    prepared.assert_not_called()
    copy.assert_not_called()
    gather.assert_not_called()
    native.assert_not_called()
    torch.testing.assert_close(values["conv_state_storage"], state_before, rtol=0, atol=0, equal_nan=True)
