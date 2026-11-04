"""A5 regression for the unfused W4A8-MXFP GMM1 used by Step activation.

Run with one idle A5 device and the CANN environment loaded:
    ASCEND_RT_VISIBLE_DEVICES=0 python test_mxfp4_unfused_gmm1.py

Weights and activations use exactly representable FP4/FP8 values so a CPU
matmul is an independent reference, without a second quantized GMM call.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_npu
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.quantization.methods.w4a8 import w4a8_mxfp4


def main():
    torch.manual_seed(11)
    torch.npu.set_device(0)
    experts, hidden, intermediate, tokens_per_expert = 2, 4096, 1280, 4
    source = torch.randint(-1, 2, (experts, 2 * intermediate, hidden), dtype=torch.int32).float() / 16
    inputs = torch.randint(-1, 2, (experts, tokens_per_expert, hidden), dtype=torch.int32).float() / 8
    reference = torch.bmm(inputs, source.transpose(1, 2)).flatten(0, 1).to(torch.bfloat16)

    def make_parameter(values):
        weight, scale = torch_npu.npu_dynamic_mx_quant(
            values.to(device="npu", dtype=torch.bfloat16), dst_type=torch_npu.float4_e2m1fn_x2
        )
        return weight.view(torch.uint8), scale.view(torch.uint8).flatten(-2)

    with torch.inference_mode():
        for expert_list in (False, True):
            layer = torch.nn.Module()
            for name, values in (
                ("w13", source),
                ("w2", torch.zeros(experts, hidden, intermediate)),
            ):
                weight, scale = make_parameter(values)
                layer.register_parameter(name + "_weight", torch.nn.Parameter(weight, requires_grad=False))
                layer.register_parameter(name + "_weight_scale", torch.nn.Parameter(scale, requires_grad=False))
            layer.activation = MoEActivation.SWIGLUSTEP
            method = object.__new__(w4a8_mxfp4.AscendW4A8MXFPDynamicFusedMoEMethod)
            method.group_size = 32
            method.dynamic_eplb = False
            method.use_expert_weight_list = expert_list
            with (
                patch.object(w4a8_mxfp4, "get_current_vllm_config", return_value=None),
                patch.object(w4a8_mxfp4, "use_cann_megamoe", return_value=False),
            ):
                method.process_weights_after_loading(layer)
            for group_list_type in (0, 1):
                counts = torch.full((experts,), tokens_per_expert, device="npu", dtype=torch.int64)
                group_list = counts.cumsum(0) if group_list_type == 0 else counts
                mlp_input = SimpleNamespace(
                    hidden_states=inputs.flatten(0, 1).to(device="npu", dtype=torch.bfloat16),
                    dynamic_scale=None,
                    layer=layer,
                    group_list=group_list,
                    group_list_type=group_list_type,
                    activation=MoEActivation.SWIGLUSTEP,
                )
                output = method.apply_gmm1(mlp_input).cpu()
                torch.testing.assert_close(output, reference, rtol=0.01, atol=1 / 128)
                print(
                    json.dumps(
                        {
                            "expert_list": expert_list,
                            "group_list_type": group_list_type,
                            "hidden": hidden,
                            "intermediate": intermediate,
                            "max_abs_error": (output.float() - reference.float()).abs().max().item(),
                        }
                    ),
                    flush=True,
                )
            del layer
            torch.npu.empty_cache()


if __name__ == "__main__":
    main()
