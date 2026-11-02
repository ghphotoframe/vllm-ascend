# SPDX-License-Identifier: Apache-2.0
"""Recognize checkpoint-native block FP8 inside a ModelSlim mixed checkpoint.

Only mutate the in-memory quantization description. Never write model files.
"""

import json
import math
import struct
from pathlib import Path


def annotate_native_block_fp8(model_dir, description):
    root = Path(model_dir)
    if not root.is_dir() or description.get("model_quant_type") != "W4A8_MXFP":
        return 0
    config = json.loads((root / "config.json").read_text())
    if "BailingMoeV3ForCausalLM" not in config.get("architectures", []):
        return 0
    indices = list(root.glob("*.safetensors.index.json"))
    if len(indices) != 1:
        raise ValueError("Expected one mixed checkpoint index")
    mapping = json.loads(indices[0].read_text())["weight_map"]
    headers = {}
    for filename in sorted(set(mapping.values())):
        path = Path(filename)
        if len(path.parts) != 1 or path.suffix != ".safetensors":
            raise ValueError("Unsafe mixed checkpoint shard path")
        with (root / path).open("rb") as f:
            length = struct.unpack("<Q", f.read(8))[0]
            if not 0 < length < (root / path).stat().st_size - 8:
                raise ValueError("Invalid safetensors header length")
            header = json.loads(f.read(length))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if name in headers or mapping.get(name) != filename:
                raise ValueError("Duplicate or unindexed mixed tensor")
            headers[name] = info
    if set(headers) != set(mapping):
        raise ValueError("Mixed checkpoint index/header mismatch")
    native = []
    for name, info in headers.items():
        if not name.endswith(".weight") or info["dtype"] != "F8_E4M3":
            continue
        if description.get(name) not in ("FLOAT", "fp8"):
            continue
        shape = info["shape"]
        scale = headers.get(name + "_scale_inv")
        if (
            len(shape) != 2
            or scale is None
            or scale["dtype"] != "F32"
            or scale["shape"] != [math.ceil(x / 128) for x in shape]
        ):
            raise ValueError(f"Invalid native block-FP8 weight/scale pair: {name}")
        native.append(name)
    # Do not leave a partially updated config after a malformed scale.
    for name in native:
        description[name] = "fp8"
    return len(native)


def configure_mixed_native_fp8(model_dir, quant_config):
    count = annotate_native_block_fp8(model_dir, quant_config.quant_description)
    if not count:
        return 0
    quant_config._mixed_native_block_fp8 = True
    quant_config.weight_block_size = (128, 128)
    install_mixed_bailing_support()
    return count


def install_mixed_bailing_support():
    # Bailing's constructors split a floating beta projection from quantized
    # q/k/v and pad native FP8 shared MLPs before TP slicing. Extend those
    # existing decisions to this mixed config without changing the vLLM repo.
    from vllm.model_executor.models import bailing_moe_v3 as bailing

    if getattr(bailing._is_block_fp8_config, "_ascend_mixed_native", False):
        return
    original_block = bailing._is_block_fp8_config
    original_excluded = bailing._is_fp8_module_excluded

    def is_block(config):
        return bool(getattr(config, "_mixed_native_block_fp8", False)) or original_block(config)

    def is_excluded(config, prefix):
        if not getattr(config, "_mixed_native_block_fp8", False):
            return original_excluded(config, prefix)
        from .modelslim_config import get_quant_type_for_layer

        config._update_packed_modules_mapping("bailing_hybrid")
        kind = get_quant_type_for_layer(
            config.quant_description,
            prefix,
            config.packed_modules_mapping,
            prefix_mapper=lambda n: config.quant_prefix_mapper("bailing_hybrid", n),
        )
        return kind != "fp8"

    is_block._ascend_mixed_native = True
    bailing._is_block_fp8_config = is_block
    bailing._is_fp8_module_excluded = is_excluded
