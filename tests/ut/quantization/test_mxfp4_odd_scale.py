# SPDX-License-Identifier: Apache-2.0
"""MXFP4 odd MX32 group counts must round-trip without losing a scale."""

from unittest.mock import patch

import pytest
import torch

from vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4 import AscendW4A8MXFPDynamicLinearMethod


@pytest.mark.parametrize("groups", [1, 4, 5, 6, 40])
def test_pair_padding_and_restore(groups):
    method = AscendW4A8MXFPDynamicLinearMethod.__new__(AscendW4A8MXFPDynamicLinearMethod)
    method.group_size = 32
    layer = torch.nn.Module()
    weight = (torch.arange(7 * groups * 16).reshape(7, groups * 16) % 256).to(torch.uint8)
    scale = (torch.arange(7 * groups).reshape(7, groups) + 120).to(torch.uint8)
    layer.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scale.clone(), requires_grad=False)
    with patch("torch_npu.npu_format_cast", side_effect=lambda x, *args, **kwargs: x.clone()):
        method.process_weights_after_loading(layer)
        assert layer.weight_scale.shape == ((groups + 1) // 2, 7, 2)
        unpacked = layer.weight_scale.transpose(0, 1).reshape(7, -1)
        assert torch.equal(unpacked[:, :groups], scale)
        if groups % 2:
            assert bool((unpacked[:, groups:] == 0).all())
        weight_once = layer.weight.clone()
        scale_once = layer.weight_scale.clone()
        method.process_weights_after_loading(layer)
        assert torch.equal(layer.weight, weight_once)
        assert torch.equal(layer.weight_scale, scale_once)
        method.restore_weights_for_rl_loading(layer)
        assert torch.equal(layer.weight, weight)
        assert torch.equal(layer.weight_scale, scale)
        method.process_weights_after_loading(layer)
        assert torch.equal(layer.weight_scale, scale_once)
