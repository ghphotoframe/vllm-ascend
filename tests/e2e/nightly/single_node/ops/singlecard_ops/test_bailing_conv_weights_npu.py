# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Real NPU graph storage survives two Ling conv checkpoint reloads."""

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
)
from vllm.model_executor.model_loader.utils import process_weights_after_loading

from tests.ut.ops.test_bailing_conv_weights import _CONVS, _expected, _new_attention


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.no_grad()
def test_graph_replay_reads_packed_conv_weights_after_two_rl_updates(dtype):
    # Real reloader metadata records the materialization device at model init.
    with torch.device("npu"):
        model, attention, config, _events = _new_attention(dtype, device="npu")
    process_weights_after_loading(model, config, torch.device("npu"))
    packed = attention._conv_weights_t(dtype)
    pointer = packed.data_ptr()
    multiplier = torch.full_like(packed, 2)

    # Capture an actual device read of every packed coefficient; repacking is
    # deliberately outside the graph and must not replace this storage.
    for _ in range(3):
        output = attention._conv_weights_t(dtype) * multiplier
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = attention._conv_weights_t(dtype) * multiplier

    for generation, order in enumerate((tuple(reversed(_CONVS)), _CONVS), start=1):
        initialize_layerwise_reload(model)
        for index, name in enumerate(order):
            weight = getattr(attention, name).weight
            value = torch.full(weight.shape, generation * 8 + index, dtype=torch.float32, device="npu")
            weight.weight_loader(weight, value)
        for name in ("A_log", "dt_bias"):
            weight = getattr(attention, name)
            weight.weight_loader(weight, torch.full(weight.shape, generation, dtype=weight.dtype, device="npu"))
        finalize_layerwise_reload(model, config)
        assert attention._conv_weights_t(dtype) is packed
        assert packed.data_ptr() == pointer
        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()
        expected = _expected(attention, dtype).cpu() * 2
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.no_grad()
def test_cpu_constructed_weights_pack_on_npu_and_return_parameters_to_cpu(dtype):
    # Exercise the real initial-load driver's device_loading_context. It moves
    # parameters only; the derived buffer must follow the children on first pack.
    with torch.device("cpu"):
        model, attention, config, _events = _new_attention(dtype, device="cpu")
    process_weights_after_loading(model, config, torch.device("npu"))
    packed = attention._conv_weights_t(dtype)
    pointer = packed.data_ptr()
    assert packed.device.type == "npu"
    assert all(param.device.type == "cpu" for param in model.parameters())
    torch.testing.assert_close(packed.cpu(), _expected(attention, dtype), rtol=0, atol=0)

    for _ in range(3):
        output = packed * 2
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = attention._conv_weights_t(dtype) * 2
    for generation in (1, 2):
        for index, name in enumerate(_CONVS):
            getattr(attention, name).weight.fill_(generation * 8 + index)
        # This is ordinary postload after CPU updates, not layerwise RL reload
        # with offload (a combination the upstream reloader does not support).
        process_weights_after_loading(model, config, torch.device("npu"))
        assert attention._conv_weights_t(dtype) is packed
        assert packed.data_ptr() == pointer
        assert all(param.device.type == "cpu" for param in model.parameters())
        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu(), _expected(attention, dtype) * 2, rtol=0, atol=0)
