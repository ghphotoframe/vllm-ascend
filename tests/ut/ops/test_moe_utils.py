import sys
import unittest
from types import ModuleType, SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock, patch

import torch
import torch_npu  # noqa: F401 -- registers torch.npu used by the module under test
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.ops.fused_moe.moe_utils import (
    _custom_gmm_swiglu_enabled,
    _get_cann_mega_moe_quant_settings,
    _pad_tokens_pair,
    _pad_tokens_with_cat,
    _prepare_dequant_swiglu_weight_scale,
    cumsum_group_list,
    load_cann_mega_moe_ops,
    select_mega_moe_activation_kwargs,
)
from vllm_ascend.quantization.quant_type import QuantType


class TestTokenPadding(unittest.TestCase):
    def test_explicit_tp_preserves_padding_without_current_config(self):
        # V2 capture can run without the initialization-only config context.
        # The caller still knows its physical TP size, including under EP.
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for tokens, padded in ((0, 8), (1, 8), (2, 8), (4, 8), (8, 8)):
                with (
                    self.subTest(dtype=dtype, tokens=tokens),
                    patch.dict("vllm_ascend.ops.fused_moe.moe_utils._PAD_ZERO_BLOCKS", {}, clear=True),
                ):
                    x = torch.arange(tokens * 12, dtype=dtype).reshape(tokens, 12)[:, ::2]
                    expected = torch.nn.functional.pad(x, (0, 0, 0, padded - tokens))
                    with (
                        patch("vllm_ascend.ops.fused_moe.moe_utils.get_current_vllm_config_or_none", return_value=None),
                        patch("torch.nn.functional.pad", side_effect=AssertionError("unexpected padding fallback")),
                    ):
                        actual = _pad_tokens_with_cat(x, padded, tp_size=8)
                    self.assertTrue(torch.equal(actual, expected))
                    actual.fill_(42)
                    replay = _pad_tokens_with_cat(x, padded, tp_size=8)
                    self.assertTrue(torch.equal(replay, expected))

    def test_tp_changes_do_not_reuse_a_shorter_zero_block(self):
        with patch.dict("vllm_ascend.ops.fused_moe.moe_utils._PAD_ZERO_BLOCKS", {}, clear=True):
            x = torch.tensor([[1.0, -2.0]])
            for tp_size in (2, 8, 4, 16, 2):
                with self.subTest(tp_size=tp_size):
                    expected = torch.nn.functional.pad(x, (0, 0, 0, tp_size - 1))
                    self.assertTrue(torch.equal(_pad_tokens_with_cat(x, tp_size, tp_size=tp_size), expected))

    def test_oversized_padding_keeps_fallback_and_two_argument_calls(self):
        x = torch.tensor([[1.0, -2.0]])
        expected = torch.nn.functional.pad(x, (0, 0, 0, 10))
        with patch("vllm_ascend.ops.fused_moe.moe_utils.get_current_vllm_config_or_none", return_value=None):
            self.assertTrue(torch.equal(_pad_tokens_with_cat(x, 11, tp_size=8), expected))
            self.assertTrue(torch.equal(_pad_tokens_with_cat(x, 11), expected))

    def test_pair_padding_preserves_values_sources_and_fresh_outputs_on_cpu(self):
        for rows in (0, 1, 2, 4, 8):
            for strided in (False, True):
                with self.subTest(rows=rows, strided=strided):
                    hidden = torch.arange(rows * 10, dtype=torch.bfloat16).reshape(rows, 10)
                    router = torch.arange(rows * 6, dtype=torch.float32).reshape(rows, 6)
                    if strided:
                        hidden, router = hidden[:, ::2], router[:, ::2]
                    originals = hidden.clone(), router.clone()
                    expected = tuple(torch.nn.functional.pad(value, (0, 0, 0, 8 - rows)) for value in originals)
                    actual = _pad_tokens_pair(hidden, router, 8, tp_size=8)
                    for output, reference, source, before in zip(actual, expected, (hidden, router), originals):
                        self.assertTrue(torch.equal(output, reference))
                        self.assertEqual(output.dtype, source.dtype)
                        self.assertTrue(torch.equal(source, before))
                        if source.numel():
                            self.assertNotEqual(output.data_ptr(), source.data_ptr())
                        output.fill_(42)
                        self.assertTrue(torch.equal(source, before))
                    replay = _pad_tokens_pair(hidden, router, 8, tp_size=8)
                    self.assertTrue(all(torch.equal(value, reference) for value, reference in zip(replay, expected)))

    @staticmethod
    def _npu_padding_input(rows, width, dtype):
        # Eligibility uses only metadata; the device kernel is isolated below.
        return SimpleNamespace(
            ndim=2,
            shape=(rows, width),
            dtype=dtype,
            device=SimpleNamespace(type="npu", index=0),
            is_contiguous=lambda: True,
        )

    def test_pair_padding_submits_both_eligible_inputs_to_one_device_call(self):
        module_name = "vllm_ascend.ops.triton.moe_token_padding"
        kernel_module = ModuleType(module_name)
        outputs = object(), object()
        kernel_module.pad_tokens_pair = MagicMock(return_value=outputs)
        for rows in (0, 1, 2, 4, 8):
            with (
                self.subTest(rows=rows),
                patch.dict(sys.modules, {module_name: kernel_module}),
                patch("vllm_ascend.ops.fused_moe.moe_utils.torch_npu.get_npu_format", return_value=2, create=True),
            ):
                hidden = self._npu_padding_input(rows, 2560, torch.bfloat16)
                router = self._npu_padding_input(rows, 512, torch.float32)
                kernel_module.pad_tokens_pair.reset_mock()
                with patch(
                    "vllm_ascend.ops.fused_moe.moe_utils._pad_tokens_with_cat",
                    side_effect=AssertionError("eligible pairs must use one device call"),
                ):
                    self.assertIs(_pad_tokens_pair(hidden, router, 8, tp_size=8), outputs)
                kernel_module.pad_tokens_pair.assert_called_once_with(hidden, router, 8)

    def test_pair_padding_keeps_single_input_fallback_for_other_contracts(self):
        cases = (
            "hidden_dtype",
            "router_dtype",
            "hidden_stride",
            "router_stride",
            "cpu",
            "device_mismatch",
            "row_mismatch",
            "rank",
            "empty_width",
            "large_batch",
            "large_padding",
            "no_tp",
            "negative_padding",
            "hidden_storage_format",
            "router_storage_format",
        )
        for case in cases:
            with self.subTest(case=case):
                hidden = self._npu_padding_input(1, 2560, torch.bfloat16)
                router = self._npu_padding_input(1, 512, torch.float32)
                padded_len, tp_size = 8, 8
                if case == "hidden_dtype":
                    hidden.dtype = torch.float16
                elif case == "router_dtype":
                    router.dtype = torch.bfloat16
                elif case == "hidden_stride":
                    hidden.is_contiguous = lambda: False
                elif case == "router_stride":
                    router.is_contiguous = lambda: False
                elif case == "cpu":
                    hidden.device = router.device = SimpleNamespace(type="cpu", index=None)
                elif case == "device_mismatch":
                    router.device = SimpleNamespace(type="npu", index=1)
                elif case == "row_mismatch":
                    router.shape = (2, 512)
                elif case == "rank":
                    router.ndim, router.shape = 3, (1, 1, 512)
                elif case == "empty_width":
                    hidden.shape = (1, 0)
                elif case == "large_batch":
                    hidden.shape, router.shape, padded_len = (9, 2560), (9, 512), 10
                elif case == "large_padding":
                    padded_len = 10
                elif case == "no_tp":
                    tp_size = None
                elif case == "negative_padding":
                    padded_len = 0
                outputs = object(), object()
                with (
                    patch(
                        "vllm_ascend.ops.fused_moe.moe_utils._pad_tokens_with_cat", side_effect=outputs
                    ) as single_pad,
                    patch(
                        "vllm_ascend.ops.fused_moe.moe_utils.torch_npu.get_npu_format",
                        side_effect=lambda value, case=case, hidden=hidden, router=router: 29
                        if (case == "hidden_storage_format" and value is hidden)
                        or (case == "router_storage_format" and value is router)
                        else 2,
                        create=True,
                    ),
                ):
                    self.assertEqual(_pad_tokens_pair(hidden, router, padded_len, tp_size=tp_size), outputs)
                self.assertEqual(single_pad.call_count, 2)
                self.assertEqual(single_pad.call_args_list[0].args, (hidden, padded_len))
                self.assertEqual(single_pad.call_args_list[1].args, (router, padded_len))
                self.assertTrue(all(call.kwargs == {"tp_size": tp_size} for call in single_pad.call_args_list))

    def test_pair_padding_retains_negative_padding_assertion(self):
        hidden = torch.ones(1, 4, dtype=torch.bfloat16)
        router = torch.ones(1, 2, dtype=torch.float32)
        with self.assertRaises(AssertionError):
            _pad_tokens_pair(hidden, router, 0, tp_size=8)


class TestCumsumGroupList(unittest.TestCase):
    glist_dict: ClassVar[dict[int, torch.Tensor]]

    @classmethod
    def setUpClass(cls):
        cls.glist_dict = {
            0: torch.tensor([0, 2, 3, 3]),
            1: torch.tensor([0, 2, 1, 0]),
            2: torch.tensor([[1, 2], [2, 1], [0, 0], [0, 0]]),
        }

    support_combine = [(0, 0), (1, 0), (0, 1)]
    unsupported_combine = [(0, 2), (2, 1), (1, 2)]

    def test_cumsum_group_list_supported_conversion(self):
        for src_list_type, dst_list_type in self.support_combine:
            with self.subTest(src=src_list_type, dst=dst_list_type):
                result = cumsum_group_list(self.glist_dict[src_list_type], src_list_type, dst_list_type, expert_num=4)
                self.assertTrue(torch.equal(result, self.glist_dict[dst_list_type]))

    def test_cumsum_group_list_invalid_type_valueerror(self):
        with self.assertRaises(ValueError) as excinfo:
            cumsum_group_list(self.glist_dict[0], 4, 0)
        self.assertIn("group_list_type should be in [0, 1, 2], but received", str(excinfo.exception))

    def test_cumsum_group_list_unsupported_conversion_notimplementederror(self):
        for src_list_type, dst_list_type in self.unsupported_combine:
            with self.subTest(src=src_list_type, dst=dst_list_type):
                with self.assertRaises(NotImplementedError) as excinfo:
                    cumsum_group_list(self.glist_dict[0], src_list_type, dst_list_type)
                self.assertIn("This feature is under development.", str(excinfo.exception))


class TestFusionFlags(unittest.TestCase):
    def test_custom_gmm_swiglu_requires_fusion_dynamic_eplb(self):
        self.assertFalse(_custom_gmm_swiglu_enabled(False, True))
        self.assertFalse(_custom_gmm_swiglu_enabled(True, False))
        with patch("vllm_ascend.ops.fused_moe.moe_utils.enable_custom_op", return_value=True):
            self.assertTrue(_custom_gmm_swiglu_enabled(True, True, activation="silu"))


class TestSwigluScaleHelpers(unittest.TestCase):
    def test_prepare_dequant_swiglu_weight_scale_stacks_and_casts(self):
        scales = [torch.randn(4, dtype=torch.float16) for _ in range(2)]
        out = _prepare_dequant_swiglu_weight_scale(scales, True)
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(out.dim(), 2)

        single = torch.randn(4, dtype=torch.float16)
        out_single = _prepare_dequant_swiglu_weight_scale([single], True)
        self.assertEqual(out_single.dtype, torch.float32)
        self.assertEqual(out_single.shape, (1, 4))

    def test_prepare_dequant_swiglu_weight_scale_keeps_flat_for_non_swigluoai(self):
        single = torch.randn(4, dtype=torch.float16)
        out = _prepare_dequant_swiglu_weight_scale([single], False)
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(out.shape, (4,))


class TestMegaMoeLoading(unittest.TestCase):
    def test_preloads_current_comm_extension(self):
        ops = SimpleNamespace(get_symm_buffer_for_mega_moe=object(), mega_moe=object())
        comm = SimpleNamespace(comm_context_op_builder=MagicMock())
        with patch("vllm_ascend.ops.fused_moe.moe_utils.import_module", side_effect=[ops, comm]):
            self.assertEqual(load_cann_mega_moe_ops(), (ops.get_symm_buffer_for_mega_moe, ops.mega_moe))
        comm.comm_context_op_builder.load.assert_called_once_with()

    def test_preloads_legacy_comm_extension(self):
        ops = SimpleNamespace(get_symm_buffer_for_mega_moe=object(), mega_moe=object())
        comm = SimpleNamespace(CommContextManager=MagicMock())
        missing = ModuleNotFoundError(name="cann_ops_transformer.ops.mc2")
        with patch("vllm_ascend.ops.fused_moe.moe_utils.import_module", side_effect=[ops, missing, comm]):
            load_cann_mega_moe_ops()
        comm.CommContextManager._ensure_loaded.assert_called_once_with()

    def test_does_not_hide_missing_dependency(self):
        missing = ModuleNotFoundError(name="a_missing_cann_dependency")
        with (
            patch("vllm_ascend.ops.fused_moe.moe_utils.import_module", side_effect=[MagicMock(), missing]),
            self.assertRaises(ModuleNotFoundError) as result,
        ):
            load_cann_mega_moe_ops()
        self.assertIs(result.exception, missing)


class TestMegaMoeQuantSettings(unittest.TestCase):
    def test_mxfp8_uses_e4m3_dispatch_and_weights(self):
        self.assertEqual(_get_cann_mega_moe_quant_settings(QuantType.W8A8MXFP), (4, 24, 24))

    def test_preserves_other_megamoe_quant_layouts(self):
        for quant_type, expected in (
            (QuantType.W8A8, (2, 258, 258)),
            (QuantType.W4A8, (2, 258, 285)),
            (QuantType.NONE, (0, None, None)),
            (QuantType.W4A8MXFP, (4, 24, 296)),
            (QuantType.W4A4MXFP, (4, 296, 296)),
        ):
            with self.subTest(quant_type=quant_type):
                self.assertEqual(_get_cann_mega_moe_quant_settings(quant_type), expected)


class TestMegaMoeActivationKwargs(unittest.TestCase):
    def test_swiglustep_binds_activation_and_layer_limit(self):
        def mega_moe(*args, activation="swiglu", activation_clamp=None, activation_params=None):
            return activation, activation_clamp, activation_params

        for activation in ("swiglustep", MoEActivation.SWIGLUSTEP):
            for clamp, expected_limit in ((None, 7.0), (0.0, 7.0), (4.0, 4.0), (10.0, 10.0)):
                with self.subTest(activation=activation, clamp=clamp):
                    kwargs = select_mega_moe_activation_kwargs(
                        mega_moe,
                        activation=activation,
                        activation_clamp=clamp,
                        swiglu_alpha=1.702,
                        swiglu_beta=1.0,
                    )
                    self.assertEqual(mega_moe(**kwargs), ("swiglustep", expected_limit, None))
                    self.assertEqual(kwargs, {"activation": "swiglustep", "activation_clamp": expected_limit})

    def test_swiglustep_does_not_require_activation_params(self):
        def mega_moe(*, activation, activation_clamp):
            return activation, activation_clamp

        kwargs = select_mega_moe_activation_kwargs(
            mega_moe, activation="swiglustep", activation_clamp=4.0, swiglu_alpha=1.0, swiglu_beta=0.0
        )
        self.assertEqual(mega_moe(**kwargs), ("swiglustep", 4.0))

    def test_swiglustep_rejects_wrappers_without_explicit_activation_and_clamp(self):
        def legacy(*, activation_clamp=None):
            """This legacy wrapper does not support activation or activation_params."""

        def oai_only(*, activation_clamp=None, glu_alpha=1.0, glu_bias=0.0, **kwargs):
            """Unlike another API with activation and activation_params, use direct scalars."""

        def no_clamp(*, activation="swiglu", activation_params=None):
            pass

        def generic(*args, **kwargs):
            pass

        for op in (legacy, oai_only, no_clamp, generic):
            with self.subTest(op=op), self.assertRaisesRegex(RuntimeError, "does not expose SwiGLU-Step"):
                select_mega_moe_activation_kwargs(
                    op, activation="swiglustep", activation_clamp=4.0, swiglu_alpha=1.0, swiglu_beta=0.0
                )

    def test_swiglustep_reads_generic_wrapper_metadata(self):
        for metadata in ("_schema", "__doc__"):
            with self.subTest(metadata=metadata):

                def mega_moe(*args, **kwargs):
                    return kwargs

                setattr(mega_moe, metadata, "mega_moe(..., str activation='swiglu', float? activation_clamp=None)")
                kwargs = select_mega_moe_activation_kwargs(
                    mega_moe,
                    activation="swiglustep",
                    activation_clamp=4.0,
                    swiglu_alpha=1.0,
                    swiglu_beta=0.0,
                )
                self.assertEqual(mega_moe(**kwargs), {"activation": "swiglustep", "activation_clamp": 4.0})

    def test_swiglustep_reads_compiled_schema(self):
        class CompiledMegaMoe:
            __signature__ = "not inspectable"
            _schema = "mega_moe(..., str activation='swiglu', float? activation_clamp=None)"

            def __call__(self, *args, **kwargs):
                return kwargs

        op = CompiledMegaMoe()
        kwargs = select_mega_moe_activation_kwargs(
            op, activation="swiglustep", activation_clamp=4.0, swiglu_alpha=1.0, swiglu_beta=0.0
        )
        self.assertEqual(op(**kwargs), {"activation": "swiglustep", "activation_clamp": 4.0})

    def test_situ_preserves_both_parameters_without_swiglu_clamping(self):
        def mega_moe(*args, activation="swiglu", activation_params=None, activation_clamp=None):
            return activation, activation_params, activation_clamp

        kwargs = select_mega_moe_activation_kwargs(
            mega_moe,
            activation="situ",
            activation_clamp=7.0,
            swiglu_alpha=1.702,
            swiglu_beta=1.0,
            situ_beta=4.0,
            situ_linear_beta=25.0,
        )
        self.assertEqual(mega_moe(**kwargs), ("situglu", {"beta": 4.0, "linear_beta": 25.0}, None))

    def test_situ_without_linear_saturation_omits_linear_beta(self):
        def mega_moe(*, activation, activation_params):
            return activation, activation_params

        kwargs = select_mega_moe_activation_kwargs(
            mega_moe, activation="situ", activation_clamp=None, swiglu_alpha=1.0, swiglu_beta=0.0
        )
        self.assertEqual(mega_moe(**kwargs), ("situglu", {"beta": 1.0}))

    def test_situ_preserves_explicit_zero_parameters(self):
        def mega_moe(*, activation, activation_params):
            return activation, activation_params

        kwargs = select_mega_moe_activation_kwargs(
            mega_moe,
            activation=MoEActivation.SITU,
            activation_clamp=4.0,
            swiglu_alpha=1.702,
            swiglu_beta=1.0,
            situ_beta=0.0,
            situ_linear_beta=0.0,
        )
        self.assertEqual(mega_moe(**kwargs), ("situglu", {"beta": 0.0, "linear_beta": 0.0}))

    def test_situ_rejects_legacy_and_oai_only_wrappers(self):
        def legacy(*args, activation_clamp=None):
            pass

        def oai_only(*args, activation_clamp=None, glu_alpha=1.0, glu_bias=0.0):
            pass

        for op in (legacy, oai_only):
            with self.subTest(op=op), self.assertRaisesRegex(RuntimeError, "does not expose SiTU"):
                select_mega_moe_activation_kwargs(
                    op, activation="situ", activation_clamp=None, swiglu_alpha=1.0, swiglu_beta=0.0
                )

    def test_other_activations_preserve_legacy_passthrough(self):
        def legacy(*, activation_clamp=None):
            return activation_clamp

        for activation in (
            "silu",
            "swiglu",
            "gelu",
            "gelu_tanh",
            "relu",
            "relu2",
            "silu_no_mul",
            "legacy_custom",
            None,
            MoEActivation.SILU,
            MoEActivation.GELU,
            MoEActivation.RELU2,
        ):
            for clamp in (None, 7.0):
                with (
                    self.subTest(activation=activation, clamp=clamp),
                    patch(
                        "vllm_ascend.ops.fused_moe.moe_utils.inspect.signature",
                        side_effect=AssertionError("legacy activation must not probe the wrapper"),
                    ),
                ):
                    kwargs = select_mega_moe_activation_kwargs(
                        legacy,
                        activation=activation,
                        activation_clamp=clamp,
                        swiglu_alpha=1.702,
                        swiglu_beta=1.0,
                    )
                    self.assertEqual(kwargs, {"activation_clamp": clamp})
                    self.assertEqual(legacy(**kwargs), clamp)

    def test_select_mega_moe_activation_kwargs_binds_swiglu_aliases(self):
        def mega_moe(*args, activation_clamp=None, swiglu_alpha=1.0, swiglu_beta=0.0):
            return args, activation_clamp, swiglu_alpha, swiglu_beta

        kwargs = select_mega_moe_activation_kwargs(
            mega_moe,
            activation="swigluoai",
            activation_clamp=7.0,
            swiglu_alpha=1.702,
            swiglu_beta=1.0,
        )
        self.assertEqual(kwargs, {"activation_clamp": 7.0, "swiglu_alpha": 1.702, "swiglu_beta": 1.0})

    def test_select_mega_moe_activation_kwargs_keeps_clamp_only_for_legacy_op(self):
        def mega_moe(*args, activation_clamp=None, **kwargs):
            return args, activation_clamp, kwargs

        kwargs = select_mega_moe_activation_kwargs(
            mega_moe,
            activation="silu",
            activation_clamp=7.0,
            swiglu_alpha=1.702,
            swiglu_beta=1.0,
        )
        self.assertEqual(kwargs, {"activation_clamp": 7.0})

    def test_select_mega_moe_activation_kwargs_binds_oai_when_supported(self):
        def mega_moe(*args, activation_clamp=None, glu_alpha=1.0, glu_bias=0.0, **kwargs):
            return args, activation_clamp, glu_alpha, glu_bias, kwargs

        kwargs = select_mega_moe_activation_kwargs(
            mega_moe,
            activation="swigluoai_uninterleave",
            activation_clamp=7.0,
            swiglu_alpha=1.702,
            swiglu_beta=1.0,
        )
        self.assertEqual(kwargs["activation_clamp"], 7.0)
        self.assertEqual(kwargs["glu_alpha"], 1.702)
        self.assertEqual(kwargs["glu_bias"], 1.0)
        self.assertNotIn("swiglu_alpha", kwargs)

    def test_select_mega_moe_activation_kwargs_binds_current_cann_api(self):
        def mega_moe(*args, activation="swiglu", activation_clamp=None, activation_params=None, **kwargs):
            return args, activation, activation_clamp, activation_params, kwargs

        kwargs = select_mega_moe_activation_kwargs(
            mega_moe,
            activation="swigluoai_uninterleave",
            activation_clamp=7.0,
            swiglu_alpha=1.702,
            swiglu_beta=1.0,
        )
        self.assertEqual(
            kwargs,
            {
                "activation": "swigluoai",
                "activation_clamp": 7.0,
                "activation_params": {"alpha": 1.702, "beta": 1.0},
            },
        )

    def test_select_mega_moe_activation_kwargs_reads_schema_fallback(self):
        class CompiledMegaMoe:
            __signature__ = "not inspectable"
            _schema = "mega_moe(..., str activation='swiglu', Dict(str, float)? activation_params=None)"

            def __call__(self, *args, **kwargs):
                return args, kwargs

        kwargs = select_mega_moe_activation_kwargs(
            CompiledMegaMoe(),
            activation="swigluoai_uninterleave",
            activation_clamp=7.0,
            swiglu_alpha=1.702,
            swiglu_beta=1.0,
        )
        self.assertEqual(kwargs["activation"], "swigluoai")
        self.assertEqual(kwargs["activation_params"], {"alpha": 1.702, "beta": 1.0})

    def test_select_mega_moe_activation_kwargs_rejects_legacy_oai(self):
        def mega_moe(*args, activation_clamp=None):
            return args, activation_clamp

        with self.assertRaisesRegex(RuntimeError, "does not expose SwiGLU-OAI"):
            select_mega_moe_activation_kwargs(
                mega_moe,
                activation="swigluoai_uninterleave",
                activation_clamp=7.0,
                swiglu_alpha=1.702,
                swiglu_beta=1.0,
            )

    def test_select_mega_moe_activation_kwargs_reads_generic_wrapper_metadata(self):
        for metadata in ("_schema", "__doc__"):
            for names, expected in (
                (
                    "activation, activation_params",
                    {"activation": "swigluoai", "activation_params": {"alpha": 1.702, "beta": 1.0}},
                ),
                ("glu_alpha, glu_bias", {"glu_alpha": 1.702, "glu_bias": 1.0}),
                ("swiglu_alpha, swiglu_beta", {"swiglu_alpha": 1.702, "swiglu_beta": 1.0}),
            ):
                with self.subTest(metadata=metadata, names=names):

                    def mega_moe(*args, **kwargs):
                        return kwargs

                    setattr(mega_moe, metadata, f"mega_moe(..., activation_clamp, {names})")
                    kwargs = select_mega_moe_activation_kwargs(
                        mega_moe,
                        activation="swigluoai",
                        activation_clamp=7.0,
                        swiglu_alpha=1.702,
                        swiglu_beta=1.0,
                    )
                    self.assertEqual(mega_moe(**kwargs), {"activation_clamp": 7.0, **expected})

    def test_select_mega_moe_activation_kwargs_preserves_explicit_signature(self):
        def mega_moe(*, activation_clamp=None, glu_alpha=1.0, glu_bias=0.0, **kwargs):
            """Unlike another API with activation and activation_params, use direct scalars."""
            self.assertFalse(kwargs)
            return activation_clamp, glu_alpha, glu_bias

        kwargs = select_mega_moe_activation_kwargs(
            mega_moe,
            activation="swigluoai",
            activation_clamp=7.0,
            swiglu_alpha=1.702,
            swiglu_beta=1.0,
        )
        self.assertEqual(mega_moe(**kwargs), (7.0, 1.702, 1.0))

    def test_select_mega_moe_activation_kwargs_rejects_metadata_for_closed_signature(self):
        def mega_moe(*, activation_clamp=None):
            """This legacy wrapper does not support activation or activation_params."""
            return activation_clamp

        with self.assertRaisesRegex(RuntimeError, "does not expose SwiGLU-OAI"):
            select_mega_moe_activation_kwargs(
                mega_moe,
                activation="swigluoai",
                activation_clamp=7.0,
                swiglu_alpha=1.702,
                swiglu_beta=1.0,
            )


if __name__ == "__main__":
    unittest.main()
