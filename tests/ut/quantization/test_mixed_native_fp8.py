# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
import struct
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parents[3] / "vllm_ascend/quantization/configs/mixed_native_fp8.py"
spec = importlib.util.spec_from_file_location("mixed_native_fp8_contract", HELPER)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
annotate = module.annotate_native_block_fp8


def checkpoint(root, *, scale_shape=(2, 2), include_scale=True, dtype="F8_E4M3"):
    names = {"model.layers.0.attention.f_proj.weight": {"dtype": dtype, "shape": [129, 256], "data_offsets": [0, 1]}}
    if include_scale:
        names["model.layers.0.attention.f_proj.weight_scale_inv"] = {
            "dtype": "F32",
            "shape": list(scale_shape),
            "data_offsets": [1, 2],
        }
    names["model.layers.0.attention.k_proj.weight"] = {"dtype": "F8_E4M3", "shape": [128, 256], "data_offsets": [2, 3]}
    names["model.layers.4.mlp.experts.0.gate_proj.weight"] = {
        "dtype": "U8",
        "shape": [128, 128],
        "data_offsets": [3, 4],
    }
    header = json.dumps(names).encode()
    (root / "weights.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"1234")
    (root / "weights.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(names, "weights.safetensors")})
    )
    (root / "config.json").write_text(json.dumps({"architectures": ["BailingMoeV3ForCausalLM"]}))
    return {
        "model_quant_type": "W4A8_MXFP",
        "model.layers.0.attention.f_proj.weight": "FLOAT",
        "model.layers.0.attention.k_proj.weight": "W8A8_MXFP8",
        "model.layers.4.mlp.experts.0.gate_proj.weight": "W4A8_MXFP",
    }


def test_native_pair_is_annotated_without_altering_mxfp_or_disk(tmp_path):
    desc = checkpoint(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert annotate(tmp_path, desc) == 1
    assert desc["model.layers.0.attention.f_proj.weight"] == "fp8"
    assert desc["model.layers.0.attention.k_proj.weight"] == "W8A8_MXFP8"
    assert desc["model.layers.4.mlp.experts.0.gate_proj.weight"] == "W4A8_MXFP"
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
    assert annotate(tmp_path, desc) == 1


@pytest.mark.parametrize("shape", [(1, 2), (2, 1), (2, 2, 1)])
def test_wrong_scale_shape_is_rejected_atomically(tmp_path, shape):
    desc = checkpoint(tmp_path, scale_shape=shape)
    before = dict(desc)
    with pytest.raises(ValueError, match="weight/scale pair"):
        annotate(tmp_path, desc)
    assert desc == before


def test_missing_scale_is_rejected(tmp_path):
    desc = checkpoint(tmp_path, include_scale=False)
    with pytest.raises(ValueError, match="weight/scale pair"):
        annotate(tmp_path, desc)


def test_bf16_float_stays_unquantized(tmp_path):
    desc = checkpoint(tmp_path, dtype="BF16")
    assert annotate(tmp_path, desc) == 0
    assert desc["model.layers.0.attention.f_proj.weight"] == "FLOAT"


def test_unrelated_model_is_untouched(tmp_path):
    desc = checkpoint(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["OtherModel"]}))
    assert annotate(tmp_path, desc) == 0


def test_unsafe_index_is_rejected(tmp_path):
    desc = checkpoint(tmp_path)
    p = tmp_path / "weights.safetensors.index.json"
    p.write_text(json.dumps({"weight_map": {"x.weight": "../weights.safetensors"}}))
    with pytest.raises(ValueError, match="Unsafe"):
        annotate(tmp_path, desc)
