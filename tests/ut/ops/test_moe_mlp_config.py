# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe import moe_comm_method
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import build_fused_experts_input
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import build_mlp_compute_input
from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import MoETokenDispatchOutput
from vllm_ascend.ops.fused_moe.moe_mlp import _unified_apply_activation
from vllm_ascend.quantization.quant_type import QuantType

MOE_MLP = "vllm_ascend.ops.fused_moe.moe_mlp"


def _config(**kwargs):
    values = dict(
        num_experts=512,
        num_local_experts=64,
        ep_size=8,
        activation=MoEActivation.SILU,
        activation_situ_beta=None,
        activation_situ_linear_beta=None,
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
    )
    values.update(kwargs)
    return SimpleNamespace(**values)


def _build_input(layer, comm_config=None, activation=MoEActivation.SILU, dtype=torch.float32):
    hidden_states = torch.tensor([[-8.0, 3.0, 8.0, -9.0, 2.0, 9.0]], dtype=dtype)
    fused = build_fused_experts_input(
        hidden_states=hidden_states,
        topk_weights=torch.ones(1, 1),
        topk_ids=torch.zeros(1, 1, dtype=torch.int32),
        layer=layer,
        quant_type=QuantType.NONE,
        dynamic_eplb=False,
        activation=activation,
    )
    dispatch = MoETokenDispatchOutput(
        hidden_states=hidden_states,
        group_list=torch.tensor([1]),
        group_list_type=1,
        combine_metadata=SimpleNamespace(),
    )
    return build_mlp_compute_input(
        fused_experts_input=fused,
        token_dispatch_output=dispatch,
        moe_config=comm_config,
        use_fusion_ops=False,
    )


class TestMoEMlpLayerConfig(unittest.TestCase):
    def test_limits_follow_layers_when_communication_objects_are_replaced(self):
        # Keep the real registry/key/setup/get and builder. Only constructing
        # hardware communication resources is replaced at the test boundary.
        def make_comm(config):
            return SimpleNamespace(moe_config=config)

        with (
            patch.dict(moe_comm_method._MoECommMethods, clear=True),
            patch.object(moe_comm_method, "AllGatherCommImpl", side_effect=make_comm),
            patch.object(moe_comm_method, "AlltoAllCommImpl", side_effect=make_comm),
            patch.object(moe_comm_method, "MC2CommImpl", side_effect=make_comm),
            patch.object(moe_comm_method, "FusedMC2CommImpl", side_effect=make_comm),
        ):
            for limits in itertools.permutations((None, 0.0, 4.0)):
                configs = [_config(swiglu_limit=limit) for limit in limits]
                layers = [SimpleNamespace(moe_config=config) for config in configs]
                for config in configs:
                    moe_comm_method.setup_moe_comm_method(config)
                self.assertEqual(len(moe_comm_method._MoECommMethods), 4)
                for comm_type in (MoECommType.ALLGATHER, MoECommType.ALLTOALL, MoECommType.MC2):
                    comm = moe_comm_method.get_moe_comm_method(comm_type, configs[-1])
                    self.assertIs(comm.moe_config, configs[-1])
                    # Revisit layers in both orders after every layer has been
                    # constructed; the result must not depend on call order.
                    for layer in layers + list(reversed(layers)):
                        with self.subTest(limits=limits, comm_type=comm_type, limit=layer.moe_config.swiglu_limit):
                            self.assertIs(moe_comm_method.get_moe_comm_method(comm_type, layer.moe_config), comm)
                            mlp = _build_input(layer, comm.moe_config)
                            self.assertEqual(mlp.swiglu_limit, layer.moe_config.swiglu_limit or 0.0)
                            self.assertIs(mlp.layer, layer)
                            self.assertIs(comm.moe_config, configs[-1])

    def test_bf16_clamping_uses_the_owning_layer(self):
        for limit in (None, 0.0, 4.0):
            with self.subTest(limit=limit):
                mlp = _build_input(
                    SimpleNamespace(moe_config=_config(swiglu_limit=limit)),
                    _config(swiglu_limit=7.0),
                    dtype=torch.bfloat16,
                )
                x = mlp.hidden_states.clone()
                gate, up = x.chunk(2, dim=-1)
                if limit:
                    gate = gate.clamp(max=limit)
                    up = up.clamp(min=-limit, max=limit)
                expected_input = torch.cat((gate, up), dim=-1)
                # Check the exact tensor presented to the existing NPU SwiGLU
                # operator, including the preserved negative gate tail.
                with patch(f"{MOE_MLP}.torch_npu.npu_swiglu", return_value="out", create=True) as swiglu:
                    self.assertEqual(_unified_apply_activation(mlp, x, None), "out")
                torch.testing.assert_close(swiglu.call_args.args[0], expected_input, rtol=0, atol=0)

    def test_swiglu_oai_parameters_follow_the_owning_layer(self):
        layer = SimpleNamespace(moe_config=_config(swiglu_limit=3.0, swiglu_alpha=1.702, swiglu_beta=0.0))
        mlp = _build_input(
            layer,
            _config(swiglu_limit=7.0, swiglu_alpha=2.0, swiglu_beta=1.0),
            activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        )
        with patch(f"{MOE_MLP}.DeviceOperator.clipped_swiglu", return_value="out") as clipped:
            self.assertEqual(_unified_apply_activation(mlp, "x", None), "out")
        clipped.assert_called_once_with("x", swiglu_limit=3.0, swiglu_alpha=1.702, swiglu_beta=0.0)

    def test_situ_parameters_follow_the_owning_layer(self):
        layer = SimpleNamespace(moe_config=_config(activation_situ_beta=4.0, activation_situ_linear_beta=25.0))
        mlp = _build_input(
            layer,
            _config(activation_situ_beta=2.0, activation_situ_linear_beta=5.0),
            activation=MoEActivation.SITU,
        )
        gate, up = mlp.hidden_states.chunk(2, dim=-1)
        expected = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate) * 25.0 * torch.tanh(up / 25.0)
        actual = _unified_apply_activation(mlp, mlp.hidden_states, None)
        torch.testing.assert_close(actual, expected)

    def test_explicit_none_uses_defaults_without_falling_back_to_shared_config(self):
        layer = SimpleNamespace(moe_config=_config())
        mlp = _build_input(
            layer,
            _config(
                swiglu_limit=4.0,
                swiglu_alpha=1.702,
                swiglu_beta=1.0,
                activation_situ_beta=4.0,
                activation_situ_linear_beta=25.0,
            ),
        )
        self.assertEqual((mlp.swiglu_limit, mlp.swiglu_alpha, mlp.swiglu_beta), (0.0, 1.0, 0.0))
        self.assertIsNone(mlp.activation_situ_beta)
        self.assertIsNone(mlp.activation_situ_linear_beta)

    def test_layer_config_works_without_a_communication_config(self):
        mlp = _build_input(SimpleNamespace(moe_config=_config(swiglu_limit=4.0)))
        self.assertEqual(mlp.swiglu_limit, 4.0)

    def test_legacy_callers_retain_explicit_config_and_defaults(self):
        for layer in (None, SimpleNamespace(), SimpleNamespace(moe_config=None)):
            with self.subTest(layer=layer):
                mlp = _build_input(layer, _config(swiglu_limit=4.0, swiglu_alpha=1.702, swiglu_beta=1.0))
                self.assertEqual((mlp.swiglu_limit, mlp.swiglu_alpha, mlp.swiglu_beta), (4.0, 1.702, 1.0))
                default = _build_input(layer)
                self.assertEqual((default.swiglu_limit, default.swiglu_alpha, default.swiglu_beta), (0.0, 1.0, 0.0))
                self.assertIsNone(default.activation_situ_beta)
                self.assertIsNone(default.activation_situ_linear_beta)


if __name__ == "__main__":
    unittest.main()
