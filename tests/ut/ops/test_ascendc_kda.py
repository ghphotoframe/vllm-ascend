#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops import ascendc_kda


class _TensorSpec:
    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device_index: int = 0,
    ):
        self.shape = shape
        self.dtype = dtype
        self.device = SimpleNamespace(type="npu", index=device_index)

    def dim(self) -> int:
        return len(self.shape)

    def numel(self) -> int:
        result = 1
        for dim in self.shape:
            result *= dim
        return result



def _supported_recurrent_specs():
    q = _TensorSpec((1, 4, 2, 128), torch.bfloat16)
    k = _TensorSpec((1, 4, 2, 128), torch.bfloat16)
    v = _TensorSpec((1, 4, 2, 128), torch.bfloat16)
    g = _TensorSpec((1, 4, 2, 128), torch.float32)
    beta = _TensorSpec((1, 4, 2), torch.float32)
    state = _TensorSpec((8, 2, 128, 128), torch.float32)
    cu_seqlens = _TensorSpec((3,), torch.int32)
    state_indices = _TensorSpec((4,), torch.int64)
    accepted = _TensorSpec((2,), torch.int32)
    return (
        q,
        k,
        v,
        g,
        beta,
        state,
        cu_seqlens,
        state_indices,
        accepted,
    )


def test_can_use_ascendc_recurrent_kda_checks_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        ascendc_kda,
        "_ascendc_recurrent_kda_runtime_available",
        lambda: True,
    )
    specs = _supported_recurrent_specs()
    assert ascendc_kda.can_use_ascendc_recurrent_kda(*specs)

    invalid_q = _TensorSpec((1, 4, 2, 128), torch.float16)
    assert not ascendc_kda.can_use_ascendc_recurrent_kda(invalid_q, *specs[1:])

    invalid_state = _TensorSpec((8, 2, 128, 256), torch.float32)
    assert not ascendc_kda.can_use_ascendc_recurrent_kda(
        *specs[:5],
        invalid_state,
        *specs[6:],
    )

    invalid_spec_indices = _TensorSpec((2, 9), torch.int32)
    assert not ascendc_kda.can_use_ascendc_recurrent_kda(
        *specs[:7],
        invalid_spec_indices,
        specs[8],
    )


def test_ascendc_recurrent_kda_uses_processed_gate_and_mutable_state_view(
    monkeypatch: pytest.MonkeyPatch,
):
    q = torch.randn(1, 2, 1, 128, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = torch.randn(1, 2, 1, 128, dtype=torch.float32)
    beta = torch.rand(1, 2, 1, dtype=torch.float32)
    state_storage = torch.zeros(4, 1, 128, 256, dtype=torch.float32)
    state = state_storage[..., :128]
    assert not state.is_contiguous()
    cu_seqlens = torch.tensor([0, 1, 2], dtype=torch.int32)
    state_indices = torch.tensor([1, 3], dtype=torch.int64)
    accepted = torch.tensor([1, 1], dtype=torch.int32)
    expected = torch.randn_like(v)
    call: dict[str, object] = {}

    def fake_recurrent(*args, **kwargs):
        call["args"] = args
        call["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(
        torch.ops._C_ascend,
        "npu_recurrent_kda",
        fake_recurrent,
        raising=False,
    )

    output, final_state = ascendc_kda.ascendc_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=state_indices,
        num_accepted_tokens=accepted,
        use_qk_l2norm_in_kernel=True,
    )

    assert output is expected
    assert final_state is state
    args = call["args"]
    kwargs = call["kwargs"]
    assert args[5] is state
    assert args[6] is cu_seqlens
    assert args[7] is state_indices
    assert kwargs["A_log"] is None
    assert kwargs["dt_bias"] is None
    assert kwargs["num_accepted_tokens"] is accepted
    assert kwargs["use_qk_l2norm_in_kernel"] is True
    assert kwargs["use_gate_in_kernel"] is False
    assert kwargs["use_beta_sigmoid_in_kernel"] is False
    assert kwargs["allow_neg_eigval"] is False
    assert kwargs["safe_gate"] is False


def test_recurrent_kda_dispatches_to_ascendc(
    monkeypatch: pytest.MonkeyPatch,
):
    tensors = [torch.empty(1) for _ in range(8)]
    expected = (torch.empty(2), tensors[5])
    call: dict[str, object] = {}

    def fake_ascendc(**kwargs):
        call.update(kwargs)
        return expected

    monkeypatch.setattr(
        ascendc_kda,
        "ascendc_recurrent_kda",
        fake_ascendc,
    )

    actual = ascendc_kda.recurrent_kda(
        q=tensors[0],
        k=tensors[1],
        v=tensors[2],
        g=tensors[3],
        beta=tensors[4],
        initial_state=tensors[5],
        cu_seqlens=tensors[6],
        ssm_state_indices=tensors[7],
    )

    assert actual is expected
    assert call["initial_state"] is tensors[5]
    assert call["cu_seqlens"] is tensors[6]
    assert call["ssm_state_indices"] is tensors[7]

