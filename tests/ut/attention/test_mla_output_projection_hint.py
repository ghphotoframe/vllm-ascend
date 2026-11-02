# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the production MLA forward boundary without loading NPU workers."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class TensorDouble:
    """Only the tensor metadata and writes used by MLA's forwarding glue."""

    dtype = "bf16"
    device = "cpu"

    def __init__(self):
        self.writes = []

    def __setitem__(self, index, value):
        self.writes.append((index, value))


def load_forward():
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/mla_v1.py"
    tree = ast.parse(source.read_text())
    impl = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendMLAImpl")
    method = next(node for node in impl.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    isolated = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    namespace = {
        "torch": SimpleNamespace(Tensor=TensorDouble, zeros=lambda *args, **kwargs: TensorDouble()),
        "_EXTRA_CTX": SimpleNamespace(num_tokens=3),
        "maybe_save_kv_layer_to_connector": Mock(),
    }
    exec(compile(ast.fix_missing_locations(isolated), str(source), "exec"), namespace)
    return namespace["forward"], namespace


class TestMLAOutputProjectionHint(unittest.TestCase):
    def test_prefill_decode_and_mixed_forward_preserve_projection_hint(self):
        for decodes, prefills in ((3, 0), (0, 3), (1, 2)):
            with self.subTest(decodes=decodes, prefills=prefills):
                forward, namespace = load_forward()
                prefill = SimpleNamespace(q_nope=1, q_pe=2, k_nope=3, k_pe=4, value=5) if prefills else None
                decoded = object() if decodes else None
                projected = object()
                projection = Mock(return_value=(projected, None))
                impl = SimpleNamespace(
                    num_heads=1,
                    v_head_dim=2,
                    use_output_gate=False,
                    pcp_enabled=False,
                    dcp_enabled=False,
                    is_pcp_decode_sharded=False,
                    enable_mlapo=False,
                    fa_quant_layer=False,
                    get_num_actual_tokens=lambda metadata: 3,
                    _decode_requires_current_kv=lambda metadata: False,
                    _prefill_requires_current_kv=lambda metadata: False,
                    _cache_requires_prolog_v3_writer=lambda cache: False,
                    _uses_strided_kv_adapter=lambda: False,
                    _mla_preprocess=Mock(return_value=(decoded, prefill)),
                    _forward_decode=Mock(return_value="decode-output"),
                    _forward_prefill=Mock(return_value="prefill-output"),
                    o_proj=projection,
                )
                metadata = SimpleNamespace(
                    num_decodes=decodes,
                    num_prefills=prefills,
                    num_decode_tokens=decodes,
                    external_flashmla=None,
                )
                cache = (SimpleNamespace(shape=(2, 4)), SimpleNamespace(shape=(2, 4)))
                output = TensorDouble()
                result = forward(impl, "layer", TensorDouble(), cache, metadata, output)
                self.assertIs(result, output)
                self.assertEqual(output.writes[-1][1], projected)
                self.assertEqual(projection.call_count, 1)
                self.assertEqual(projection.call_args.kwargs, {"is_prefill": bool(prefills)})
                self.assertEqual(impl._forward_decode.call_count, bool(decodes))
                self.assertEqual(impl._forward_prefill.call_count, bool(prefills))
                namespace["maybe_save_kv_layer_to_connector"].assert_called_once_with("layer", list(cache))


if __name__ == "__main__":
    unittest.main()
