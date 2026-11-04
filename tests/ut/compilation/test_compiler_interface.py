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
# This file is a part of the vllm-ascend project.
#

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch_npu
import vllm.config.vllm as vllm_config_module
from vllm.config import get_current_vllm_config

from vllm_ascend.compilation import compiler_interface


@pytest.fixture
def compiler_runtime(monkeypatch):
    compilation_config = SimpleNamespace(
        enable_npugraph_ex=True,
        enable_static_kernel=True,
        enable_super_kernel=True,
    )
    monkeypatch.setattr(
        compiler_interface,
        "get_ascend_config",
        lambda: SimpleNamespace(ascend_compilation_config=compilation_config),
    )
    initialize_runtime = Mock()
    monkeypatch.setattr(compiler_interface, "_initialize_static_kernel_runtime", initialize_runtime)
    # CPU UTs may supply a minimal torch_npu stub without version metadata.
    monkeypatch.setattr(torch_npu, "__version__", "test-version", raising=False)
    # Exercise the real global getter, including any old imported alias. Merely
    # mocking vllm.config.get_current_vllm_config would miss that alias.
    monkeypatch.setattr(vllm_config_module, "_current_vllm_config", None)
    return compilation_config, initialize_runtime


@pytest.mark.parametrize("disable_cache", [False, True], ids=["cache-enabled", "cache-disabled"])
def test_initialize_static_runtime_uses_owning_config_without_global_context(compiler_runtime, tmp_path, disable_cache):
    _, initialize_runtime = compiler_runtime
    owner_config = SimpleNamespace()
    compiler = compiler_interface.AscendCompiler()
    compiler.compute_hash(owner_config)

    with pytest.raises(AssertionError, match="Current vLLM config is not set"):
        get_current_vllm_config()

    compiler.initialize_cache(str(tmp_path), disable_cache=disable_cache)

    initialize_runtime.assert_called_once_with(owner_config)
    assert initialize_runtime.call_args.args[0] is owner_config
    assert compiler.cache_dir == str(tmp_path)
    assert compiler.disable_cache is disable_cache
    # Initializing the compiler must not establish a process-global context.
    with pytest.raises(AssertionError, match="Current vLLM config is not set"):
        get_current_vllm_config()


def test_initialize_cache_skips_runtime_when_static_kernel_is_disabled(compiler_runtime, tmp_path):
    compilation_config, initialize_runtime = compiler_runtime
    compilation_config.enable_static_kernel = False
    compilation_config.enable_super_kernel = False
    compiler = compiler_interface.AscendCompiler()
    compiler.compute_hash(SimpleNamespace())

    compiler.initialize_cache(str(tmp_path))

    initialize_runtime.assert_not_called()
    assert compiler.cache_dir == str(tmp_path)
    assert compiler.disable_cache is False


def test_initialize_static_runtime_does_not_use_another_models_context(compiler_runtime, monkeypatch, tmp_path):
    _, initialize_runtime = compiler_runtime
    owner_config = SimpleNamespace()
    compiler = compiler_interface.AscendCompiler()
    compiler.compute_hash(owner_config)
    other_config = SimpleNamespace()
    monkeypatch.setattr(vllm_config_module, "_current_vllm_config", other_config)

    compiler.initialize_cache(str(tmp_path))

    assert initialize_runtime.call_args.args[0] is owner_config
    assert get_current_vllm_config() is other_config
