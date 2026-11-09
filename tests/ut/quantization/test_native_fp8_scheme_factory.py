# SPDX-License-Identifier: Apache-2.0
from unittest.mock import Mock, patch

import pytest

from vllm_ascend.quantization.configs import modelslim_config as module


@pytest.mark.parametrize("layer_type", ["linear", "moe", "attention"])
@pytest.mark.parametrize("quant_type", ["fp8", "W4A8_MXFP"])
def test_scheme_factory_preserves_constructor_contract(layer_type, quant_type):
    factory = Mock()
    moe_config = object()
    with patch.object(module, "get_scheme_class", return_value=factory):
        result = module.create_scheme_for_layer(quant_type, "layer", layer_type, moe_config=moe_config)
    assert result is factory.return_value
    if quant_type == "fp8" and layer_type == "linear":
        factory.assert_called_once_with((128, 128))
    elif quant_type == "fp8" and layer_type == "moe":
        factory.assert_called_once_with((128, 128), moe_config)
    else:
        factory.assert_called_once_with()


def test_missing_native_scheme_fails_before_method_creation():
    with (
        patch.object(module, "get_scheme_class", return_value=None),
        pytest.raises(NotImplementedError, match="quant_type=fp8"),
    ):
        module.create_scheme_for_layer("fp8", "layer", "linear")


@pytest.mark.parametrize("mapper_already_applied", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_cold_worker_installs_before_projection_or_mapper_early_return(mapper_already_applied, mixed):
    config = module.AscendModelSlimConfig.__new__(module.AscendModelSlimConfig)
    mapper = Mock()
    mapper.apply_dict.return_value = {"mapped.weight": "fp8"}
    config._mixed_native_block_fp8 = mixed
    config._mapper_applied = mapper_already_applied
    config.hf_to_vllm_mapper = mapper if mapper_already_applied else None
    config.quant_description = {"weight": "fp8", "optional": {"recipe": "unchanged"}}
    config._add_kvcache_quant_metadata = Mock()
    with patch.object(module, "install_mixed_bailing_support") as install:
        config.apply_vllm_mapper(mapper)
    assert install.call_count == int(mixed)
    if mapper_already_applied:
        mapper.apply_dict.assert_not_called()
    else:
        mapper.apply_dict.assert_called_once_with({"weight": "fp8"})
        assert config.quant_description["optional"] == {"recipe": "unchanged"}
        config._add_kvcache_quant_metadata.assert_called_once()
