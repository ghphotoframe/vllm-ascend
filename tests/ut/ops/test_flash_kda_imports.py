# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only import/distribution contracts for the bundled optional operator.

Fresh isolated interpreters avoid depending on an external flash_kda directory
or a developer's installed PyPTO. Hardware/bootstrap boundaries are faked; the
actual internal package/implementation/adapter sources are imported intact.
These tests do not validate NPU compilation, kernels or numerical accuracy.
"""

import ast
import fnmatch
import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_INTERNAL_MODULE = "vllm_ascend.ops.pypto.flash_kda"

_BOOTSTRAP = r"""
import builtins
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

root = Path(sys.argv[1])
def package(name, path):
    value = ModuleType(name)
    value.__path__ = [str(path)]
    value.__spec__ = importlib.util.spec_from_loader(name, loader=None, is_package=True)
    sys.modules[name] = value
    return value

# Skip Linux/vLLM initialization, but let the real optional subpackage and
# operator module resolve from their normal repository package locations.
package('vllm_ascend', root / 'vllm_ascend')
package('vllm_ascend.ops', root / 'vllm_ascend/ops')
attempts = []
original_import = builtins.__import__
def reject_external(name, *args, **kwargs):
    if name == 'flash_kda' or name.startswith('flash_kda.'):
        raise AssertionError('external flash_kda package required: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = reject_external
"""


def _isolated(script):
    result = subprocess.run(
        [sys.executable, "-I", "-c", _BOOTSTRAP + script, str(_ROOT)],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def test_optional_package_init_does_not_import_pypto_or_npu_runtime():
    actual = _isolated(r"""
def forbid_optional(name, *args, **kwargs):
    if name.split('.', 1)[0] in ('pypto', 'pypto_pro', 'torch_npu'):
        attempts.append(name)
        raise AssertionError('eager optional runtime import: ' + name)
    return reject_external(name, *args, **kwargs)
builtins.__import__ = forbid_optional
value = importlib.import_module('vllm_ascend.ops.pypto')
assert Path(value.__file__).resolve() == (root / 'vllm_ascend/ops/pypto/__init__.py').resolve()
assert 'vllm_ascend.ops.pypto.flash_kda' not in sys.modules
print(json.dumps(dict(optional_import_attempts=attempts)))
""")
    assert actual["optional_import_attempts"] == []


def test_bundled_module_imports_without_external_operator_path():
    actual = _isolated(r"""
class SymbolicModule(ModuleType):
    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return MagicMock(name=name)

torch = SymbolicModule('torch')
torch.Tensor = type('Tensor', (), {})
sys.modules['torch'] = torch
sys.modules['torch_npu'] = ModuleType('torch_npu')
pto = package('pypto_pro', Path('/not-an-installed-pypto'))
language = SymbolicModule('pypto_pro.language')
language.__path__ = []
def identity_decorator(function=None, **kwargs):
    return (lambda function: function) if function is None else function
language.vector_function = identity_decorator
language.jit = identity_decorator
language.Vf = MagicMock(name='Vf')
sys.modules[language.__name__] = language
package('pypto_pro.runtime', Path('/not-an-installed-pypto/runtime'))
platform = ModuleType('pypto_pro.runtime.platform')
def unexpected_platform_query():
    raise AssertionError('device query during operator import')
platform.get_platform_info = unexpected_platform_query
sys.modules[platform.__name__] = platform
tiling = ModuleType('pypto_pro.runtime.tilingkey')
class TilingKeyField:
    def __init__(self, **kwargs):
        self.options = kwargs
tiling.TilingKeyField = TilingKeyField
sys.modules[tiling.__name__] = tiling

value = importlib.import_module('vllm_ascend.ops.pypto.flash_kda')
assert callable(value.flash_kda_wrapper)
assert Path(value.__file__).resolve() == (root / 'vllm_ascend/ops/pypto/flash_kda.py').resolve()
assert not any(name == 'flash_kda' or name.startswith('flash_kda.') for name in sys.modules)
print(json.dumps(dict(origin=value.__file__, external_operator_loaded=False)))
""")
    assert Path(actual["origin"]).resolve() == _ROOT / "vllm_ascend/ops/pypto/flash_kda.py"
    assert not actual["external_operator_loaded"]


@pytest.mark.parametrize("missing", ["pypto", "pypto_pro"])
def test_actual_internal_import_failure_reports_missing_pypto(missing):
    actual = _isolated(
        r"""
torch = ModuleType('torch')
torch.Tensor = type('Tensor', (), {})
sys.modules['torch'] = torch
sys.modules['torch_npu'] = ModuleType('torch_npu')
device = ModuleType('vllm_ascend.device')
device.HardwareCapability = SimpleNamespace(FLASH_KDA_PREFILL=object())
device.get_current_hardware_profile = lambda: SimpleNamespace(supports=lambda capability: True)
sys.modules[device.__name__] = device
def missing_pypto(name, *args, **kwargs):
    if name.split('.', 1)[0] in ('pypto', 'pypto_pro'):
        attempts.append(name)
        raise ModuleNotFoundError('PyPTO missing in isolated interpreter', name=MISSING)
    return reject_external(name, *args, **kwargs)
builtins.__import__ = missing_pypto
adapter = importlib.import_module('vllm_ascend.ops.flash_kda')
assert attempts == [], attempts
# This test isolates the real lazy import/error boundary; shape/numeric
# contracts are covered by test_flash_kda.py using real CPU Tensor arithmetic.
adapter._validate_inputs = lambda *args, **kwargs: None
try:
    adapter.run_flash_chunk_kda(None, None, None, None, None, None, (0, 1), (), None, None, lower_bound=-5.0)
except RuntimeError as error:
    assert 'PyPTO' in str(error), str(error)
    assert 'vendor flash_kda/' not in str(error), str(error)
    assert isinstance(error.__cause__, ModuleNotFoundError)
    assert error.__cause__.name == MISSING
    frames = traceback.extract_tb(error.__cause__.__traceback__)
    internal_path = (root / 'vllm_ascend/ops/pypto/flash_kda.py').resolve()
    assert any(Path(frame.filename).resolve() == internal_path for frame in frames)
else:
    raise AssertionError('missing dependency silently succeeded or fell back')
assert len(attempts) == 1, attempts
print(json.dumps(dict(missing=MISSING, import_attempts=attempts, imported_adapter_lazily=True)))
""".replace("MISSING", repr(missing))
    )
    assert actual["missing"] == missing
    assert actual["imported_adapter_lazily"]


def test_internal_operator_package_and_licenses_are_discoverable_for_distribution():
    from setuptools import find_packages

    packages = find_packages(where=str(_ROOT), exclude=("docs", "examples", "tests*", "csrc"))
    assert "vllm_ascend.ops.pypto" in packages
    packaged = _ROOT / "vllm_ascend/ops/pypto"
    for filename in ("__init__.py", "flash_kda.py", "LICENSE", "README.md"):
        assert (packaged / filename).is_file()
    assert "CANN Open Software License" in (packaged / "LICENSE").read_text()
    setup = ast.parse((_ROOT / "setup.py").read_text())
    declaration = next(
        keyword.value
        for node in ast.walk(setup)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "setup"
        for keyword in node.keywords
        if keyword.arg == "package_data"
    )
    patterns = ast.literal_eval(declaration)["vllm_ascend.ops.pypto"]
    for filename in ("LICENSE", "README.md"):
        assert any(fnmatch.fnmatchcase(filename, pattern) for pattern in patterns)
