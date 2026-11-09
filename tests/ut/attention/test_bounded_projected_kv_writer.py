"""CPU regressions for projected-KV admission and bounded cache writes.

Load the production methods without importing the NPU plugin, so this contract
can also be checked in a dedicated container before loading model weights.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


def load_methods(source):
    names = {
        "_uses_strided_kv_adapter",
        "_uses_bounded_projected_kv_writer",
        "supports_strided_mla_writer",
        "_prepare_strided_kv_workspace",
        "_exec_strided_kv_writer",
        "_validate_prolog_v3_cache",
        "_validate_context_prolog_v3_cache",
    }
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendMLAImpl")
    cls.bases, cls.decorator_list = [], []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.Module(body=[future, cls], type_ignores=[])
    namespace = {
        "PROLOG_V3_SUPPORTED_INPUT_SIZES": frozenset((1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192)),
        "torch": torch,
        "torch_npu": SimpleNamespace(npu_kv_rmsnorm_rope_cache=object()),
        "cdiv": lambda a, b: (a + b - 1) // b,
        "DeviceOperator": SimpleNamespace(),
    }
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace["AscendMLAImpl"], namespace


SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/mla_v1.py"


class TestProjectedKVWriter(unittest.TestCase):
    def setUp(self):
        cls, self.namespace = load_methods(SOURCE)
        self.impl = cls()
        attrs = dict(
            layer_name="test-draft",
            support_fp8_attention=True,
            is_draft_model=True,
            pcp_enabled=False,
            dcp_enabled=False,
            num_kv_heads=1,
            use_mla_rope=True,
            qk_rope_head_dim=2,
            kv_lora_rank=4,
            dtype=torch.bfloat16,
            fa_quant_layer=False,
            enable_kv_nz=False,
            fused_qkv_a_proj=None,
            q_proj=None,
            _strided_kv_workspace=None,
            kv_a_layernorm=SimpleNamespace(weight=torch.ones(4, dtype=torch.bfloat16), variance_epsilon=1e-6),
            vllm_config=SimpleNamespace(
                scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
                compilation_config=SimpleNamespace(cudagraph_capture_sizes=[4]),
                cache_config=SimpleNamespace(block_size=4),
            ),
        )
        self.impl.__dict__.update(attrs)
        self.impl.supports_prolog_v3_writer = lambda: False

    def test_separate_and_mixed_projections_do_not_exclude_kv_writer(self):
        self.assertTrue(self.impl._uses_bounded_projected_kv_writer())
        self.impl.fused_qkv_a_proj = SimpleNamespace(input_size=4096)
        self.impl.q_proj = SimpleNamespace(quant_method=object())
        self.assertTrue(self.impl._uses_bounded_projected_kv_writer())

    def test_runtime_fallback_does_not_advertise_a_new_pd_layout(self):
        self.impl.fused_qkv_a_proj = SimpleNamespace(input_size=4096)
        self.impl.q_proj = SimpleNamespace(quant_method=object())
        self.impl._supports_strided_mla_projection = lambda: False
        self.assertTrue(self.impl._uses_bounded_projected_kv_writer())
        self.assertFalse(self.impl.supports_strided_mla_writer())
        self.impl._prepare_strided_kv_workspace()
        self.assertFalse(self.impl.supports_strided_mla_writer())

    def test_empty_input_does_not_touch_history(self):
        self.impl._prepare_strided_kv_workspace()
        backing = torch.full((2, 4, 1, 6), -7, dtype=torch.bfloat16)
        original = backing.clone()
        result = self.impl._exec_strided_kv_writer(
            torch.empty(0, 6), None, None, (backing[..., :4], backing[..., 4:]), torch.empty(0, dtype=torch.int64)
        )
        self.assertEqual(result[0].shape[0], 0)
        torch.testing.assert_close(backing, original, rtol=0, atol=0)

    def test_unsupported_contracts_keep_existing_paths(self):
        for key, value in [
            ("support_fp8_attention", False),
            ("is_draft_model", False),
            ("pcp_enabled", True),
            ("dcp_enabled", True),
            ("num_kv_heads", 2),
            ("use_mla_rope", False),
            ("qk_rope_head_dim", 0),
            ("dtype", torch.float16),
            ("fa_quant_layer", True),
            ("enable_kv_nz", True),
            ("kv_a_layernorm", None),
            ("kv_a_layernorm", SimpleNamespace(weight=None)),
        ]:
            with self.subTest(key=key):
                old = getattr(self.impl, key)
                setattr(self.impl, key, value)
                self.assertFalse(self.impl._uses_bounded_projected_kv_writer())
                setattr(self.impl, key, old)
        self.impl.supports_prolog_v3_writer = lambda: True
        self.assertFalse(self.impl._uses_bounded_projected_kv_writer())

    def test_workspace_needs_no_fused_projection_and_addresses_are_stable(self):
        self.impl._prepare_strided_kv_workspace()
        work = self.impl._strided_kv_workspace
        self.assertEqual(work[0].shape, (2, 4, 1, 4))
        self.assertEqual(work[2].numel(), 8)
        pointers = tuple(x.data_ptr() for x in work)
        self.impl._prepare_strided_kv_workspace()
        self.assertEqual(tuple(x.data_ptr() for x in self.impl._strided_kv_workspace), pointers)

    def test_only_current_slots_are_written_in_token_fused_cache(self):
        self.impl._prepare_strided_kv_workspace()
        history = torch.full((6, 4, 1, 6), -7, dtype=torch.bfloat16)
        cache = (history[..., :4], history[..., 4:])
        original = history.clone()
        writes = []

        def writer(x, weight, cos, sin, slots, rope, nope, **kwargs):
            self.assertTrue(rope.is_contiguous() and nope.is_contiguous())
            self.assertLessEqual(nope.numel(), 8 * 4)
            values = x.reshape(-1, 6)
            nope.flatten(0, 1)[: len(values)].copy_(values[:, :4].unsqueeze(1))
            rope.flatten(0, 1)[: len(values)].copy_(values[:, 4:].unsqueeze(1))

        def scatter(**kwargs):
            for i, slot in enumerate(kwargs["slot_mapping"].tolist()):
                if slot < 0:
                    continue
                for src, dst in [("key", "key_cache"), ("value", "value_cache")]:
                    kwargs[dst][slot // 4, slot % 4].copy_(kwargs[src][i])
                writes.append(slot)

        self.namespace["torch_npu"].npu_kv_rmsnorm_rope_cache = writer
        self.namespace["DeviceOperator"].reshape_and_cache = scatter
        x = torch.arange(18, dtype=torch.bfloat16).reshape(3, 6)
        slots = torch.tensor([1, -1, 17], dtype=torch.int64)
        pointers = tuple(v.data_ptr() for v in cache)
        self.impl._exec_strided_kv_writer(x, torch.ones(3, 2), torch.zeros(3, 2), cache, slots)
        original[0, 1, 0] = x[0]
        original[4, 1, 0] = x[2]
        torch.testing.assert_close(history, original)
        self.assertEqual(writes, [1, 17])
        self.assertEqual(tuple(v.data_ptr() for v in cache), pointers)

    def test_bad_slot_metadata_and_quantized_cache_fail_before_operator(self):
        self.impl._prepare_strided_kv_workspace()
        backing = torch.zeros(2, 4, 1, 6, dtype=torch.bfloat16)
        cache = (backing[..., :4], backing[..., 4:])
        with self.assertRaisesRegex(RuntimeError, "INT32/INT64"):
            self.impl._exec_strided_kv_writer(torch.zeros(2, 6), None, None, cache, torch.zeros(2))
        quantized_backing = torch.zeros(2, 4, 1, 6, dtype=torch.uint8)
        quantized = (quantized_backing[..., :4], quantized_backing[..., 4:])
        with self.assertRaisesRegex(RuntimeError, "unquantized caches"):
            slots = torch.zeros(2, dtype=torch.int64)
            self.impl._exec_strided_kv_writer(torch.zeros(2, 6), None, None, quantized, slots)
