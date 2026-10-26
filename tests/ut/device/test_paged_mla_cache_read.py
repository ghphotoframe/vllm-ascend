# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


class DeviceTensor:
    """Keep real CPU storage and emulate the device boundary without creating an NPU context."""

    device = SimpleNamespace(type="npu")

    def __init__(self, tensor):
        self.tensor = tensor

    def __getattr__(self, name):
        return getattr(self.tensor, name)

    def __getitem__(self, index):
        return self.tensor[index]


def load_reader(copy, fmt=2, function="try_load_strided_mla_cache"):
    # Execute the production function, isolating only device factories and the
    # final copy kernel without duplicating the slot computation.
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/device/paged_mla_cache_read.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    body: list[ast.stmt] = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.Assign))]
    cpu_torch = SimpleNamespace(
        float16=torch.float16,
        bfloat16=torch.bfloat16,
        int64=torch.int64,
        arange=lambda *args, **kwargs: torch.arange(*args, **{**kwargs, "device": "cpu"}),
        repeat_interleave=torch.repeat_interleave,
        cumsum=torch.cumsum,
        cat=torch.cat,
    )
    namespace = {
        "torch": cpu_torch,
        "torch_npu": SimpleNamespace(get_npu_format=lambda cache: fmt),
        "copy_pcp_kv_cache": copy,
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source), "exec"), namespace)
    return namespace[function]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("layout", ["component_major", "token_fused"])
@pytest.mark.parametrize(
    "slot_values",
    [
        [],
        [0],
        [-1],
        [-1, -1, -1],
        [127, 128, 255, -1, 129],
        [257, 0, 128, 128, 127, 383, 2, 255, 129],
        [-1, 127, 128, 383, 0, 128, 1, 255],
    ],
)
def test_slot_reads_preserve_values_order_and_source(dtype, layout, slot_values):
    block_size, blocks, key_dim, rope_dim = 128, 3, 512, 64
    page_elements = block_size * (key_dim + rope_dim)
    prefix_elements = 256
    raw = (torch.arange(prefix_elements + blocks * page_elements) % 127).to(dtype)
    if layout == "component_major":
        key_stride, rope_stride = key_dim, rope_dim
        rope_offset = prefix_elements + block_size * key_dim
    else:
        key_stride = rope_stride = key_dim + rope_dim
        rope_offset = prefix_elements + key_dim
    key = torch.as_strided(
        raw, (blocks, block_size, 1, key_dim), (page_elements, key_stride, key_stride, 1), prefix_elements
    )
    rope = torch.as_strided(
        raw, (blocks, block_size, 1, rope_dim), (page_elements, rope_stride, rope_stride, 1), rope_offset
    )
    before = raw.clone()
    # Supply non-contiguous, non-flat metadata to exercise slot normalization.
    slot_storage = torch.full((len(slot_values), 2), -999, dtype=torch.int32)
    slot_storage[:, 0] = torch.tensor(slot_values, dtype=torch.int32)
    slots = slot_storage[:, :1]
    slots_before = slot_storage.clone()

    class GuardedCache(DeviceTensor):
        def __getitem__(self, index):
            pytest.fail("The source cache must be read by the stride-aware kernel")

        def contiguous(self):
            pytest.fail("The source cache must not be materialized as contiguous")

    cache = (GuardedCache(key), GuardedCache(rope))
    calls = []

    def reference_row(slot):
        if slot < 0:
            return torch.zeros(key_dim + rope_dim, dtype=dtype)
        block, offset = divmod(slot, block_size)
        return torch.cat((key[block, offset, 0], rope[block, offset, 0]))

    def copy(actual_cache, actual_slots):
        assert actual_cache[0] is cache[0] and actual_cache[1] is cache[1]
        assert actual_slots.ndim == 1 and actual_slots.dtype == torch.int64
        calls.append(actual_slots.tolist())
        if not actual_slots.numel():
            return torch.empty(0, key_dim + rope_dim, dtype=dtype)
        return torch.stack([reference_row(slot) for slot in actual_slots.tolist()])

    packed = load_reader(copy, function="load_mla_cache_rows_by_slots")(*cache, slots)
    expected = (
        torch.stack([reference_row(slot) for slot in slot_values])
        if slot_values
        else raw.new_empty((0, key_dim + rope_dim))
    )
    torch.testing.assert_close(packed, expected, rtol=0, atol=0)
    assert packed.is_contiguous()
    if slot_values:
        assert calls == [slot_values + [slot_values[0]] * ((-len(slot_values)) % 8)]
    else:
        assert all(not call for call in calls)
    assert torch.equal(raw, before)
    assert torch.equal(slot_storage, slots_before)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("component", ["rope", "duplicate", "empty"])
@pytest.mark.parametrize("block_size", [128, 384])
@pytest.mark.parametrize("layout", ["component_major", "token_fused"])
@pytest.mark.parametrize("case", ["full_pages", "mixed_empty", "unaligned_offsets", "all_empty", "no_offset"])
def test_strided_token_reads_preserve_request_order_and_source(dtype, component, block_size, layout, case):
    lengths, starts = {
        "full_pages": ([3 * block_size], [0]),
        "mixed_empty": ([3 * block_size, block_size + 1, 0, 65], [0, block_size, 999999, 2 * block_size]),
        "unaligned_offsets": ([2, block_size + 3, 0, 2 * block_size + 3], [block_size - 1, block_size + 1, 999999, 7]),
        "all_empty": ([0], [999999]),
        "no_offset": ([block_size + 1, 0, 3], [0, 999999, 0]),
    }[case]
    blocks, key_dim, rope_dim = 15, 512, 64
    slot_elements = block_size * (key_dim + rope_dim) + 256
    prefix_elements = 256
    raw = torch.zeros(prefix_elements + blocks * slot_elements, dtype=dtype)
    if layout == "component_major":
        key_stride, rope_stride = key_dim, rope_dim
        rope_offset = prefix_elements + block_size * key_dim
    else:
        key_stride = rope_stride = key_dim + rope_dim
        rope_offset = prefix_elements + key_dim
    key = torch.as_strided(
        raw, (blocks, block_size, 1, key_dim), (slot_elements, key_stride, key_stride, 1), prefix_elements
    )
    rope = torch.as_strided(
        raw,
        (blocks, block_size, 1, rope_dim),
        (slot_elements, rope_stride, rope_stride, 1),
        rope_offset,
    )
    key.copy_((torch.arange(key.numel()).reshape(key.shape) % 127).to(dtype))
    rope.copy_((torch.arange(rope.numel()).reshape(rope.shape) % 97 + 5).to(dtype))
    value = key if component == "duplicate" else rope[..., :0] if component == "empty" else rope
    before = raw.clone()
    table_rows = [[3, 4, 5, 6, 7, 8], [3, 4, 12, 13, 14, 6], [8, 2, 10, 7, 1, 4], [3, 4, 5, 6, 7, 8]]
    table = torch.tensor([row if length else [999999] * 6 for row, length in zip(table_rows, lengths)])
    expected_slots = [
        int(table[row, (start + token) // block_size]) * block_size + (start + token) % block_size
        for row, (length, start) in enumerate(zip(lengths, starts))
        for token in range(length)
    ]
    calls = []

    def copy(cache, slots):
        calls.append(slots.tolist())
        # Use an independent row-by-row reference without the production
        # function's repeat_interleave or cumulative prefix computation.
        return torch.stack(
            [
                torch.cat([part.tensor[slot // block_size, slot % block_size, 0] for part in cache])
                for slot in slots.tolist()
            ]
        )

    tokens = sum(lengths)
    output_k = torch.empty(tokens, 1, key_dim, dtype=dtype)
    output_v = torch.empty(tokens, 1, value.shape[-1], dtype=dtype)

    class GuardedCache(DeviceTensor):
        def __getitem__(self, index):
            pytest.fail("The source cache must be read by the stride-aware kernel")

        def contiguous(self):
            pytest.fail("The source cache must not be materialized as contiguous")

    # Slice metadata to preserve coverage for non-contiguous input lengths.
    lengths_storage = torch.tensor([[length, -1] for length in lengths], dtype=torch.int32)
    cache_k = GuardedCache(key)
    cache_v = cache_k if component == "duplicate" else GuardedCache(value)
    assert load_reader(copy)(
        cache_k,
        cache_v,
        DeviceTensor(table),
        DeviceTensor(lengths_storage[:, 0]),
        None if case == "no_offset" else DeviceTensor(torch.tensor(starts)),
        DeviceTensor(output_k),
        DeviceTensor(output_v),
    )
    if tokens:
        assert calls[0][:tokens] == expected_slots
        assert len(calls[0]) % 8 == 0
        assert calls[0][tokens:] == [expected_slots[0]] * ((-tokens) % 8)
        for row, slot in enumerate(expected_slots):
            assert torch.equal(output_k[row, 0], key[slot // block_size, slot % block_size, 0])
            assert torch.equal(output_v[row, 0], value[slot // block_size, slot % block_size, 0])
    else:
        assert not calls
    assert torch.equal(raw, before)


@pytest.mark.parametrize("unsupported", ["contiguous", "nz", "heads", "int8"])
@pytest.mark.parametrize("block_size", [128, 384])
def test_unaffected_layouts_keep_native_reader(unsupported, block_size):
    cache = torch.zeros(2, block_size, 1, 16, dtype=torch.bfloat16)
    if unsupported == "int8":
        cache = cache.to(torch.int8)
    if unsupported != "contiguous":
        cache = cache[..., ::2]
    if unsupported == "heads":
        cache = cache.expand(2, block_size, 2, 8)

    def forbidden_copy(*args):
        pytest.fail("Unaffected layouts must not enter the new copy path")

    fmt = 29 if unsupported == "nz" else 2
    assert not load_reader(forbidden_copy, fmt)(DeviceTensor(cache), DeviceTensor(cache), None, None, None, None, None)
