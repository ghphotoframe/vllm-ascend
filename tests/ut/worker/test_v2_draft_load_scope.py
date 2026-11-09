# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only regression for the config seen by a V2 draft constructor.

Execute the production method with a loader boundary double, rather than
initializing Torch/NPU workers. The double observes the same VllmConfig that
upstream load_eagle_model forwards to initialize_model.
"""

import ast
import copy
import pathlib
import types
import unittest

SOURCE = pathlib.Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/spec_decode/autoregressive/speculator.py"



class LoaderBoundary:
    def load_draft_model(self, target_model, layer_names):
        self.observed_config = self.vllm_config
        self.observed_target = target_model
        self.observed_names = layer_names
        if self.loader_error:
            raise RuntimeError("loader failed")
        return self.loaded_model

    def _maybe_remove_d2t(self, model):
        self.after_load_config = self.vllm_config
        self.after_load_model = model


def load_production_method():
    module = ast.parse(SOURCE.read_text())
    cls = next(x for x in module.body if isinstance(x, ast.ClassDef) and x.name == "AscendAutoRegressiveSpeculator")
    method = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "load_draft_model")
    wrapper = ast.ClassDef(
        name="Subject",
        bases=[ast.Name(id="LoaderBoundary", ctx=ast.Load())],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    isolated = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), wrapper],
        type_ignores=[],
    )
    namespace = {"LoaderBoundary": LoaderBoundary, "copy": copy.copy}
    exec(compile(ast.fix_missing_locations(isolated), str(SOURCE), "exec"), namespace)
    return namespace["Subject"]


class DraftLoadScopeTest(unittest.TestCase):
    def setUp(self):
        self.original_model_config = types.SimpleNamespace(runner_type="generate", hf_config=object())
        self.original_config = types.SimpleNamespace(
            model_config=self.original_model_config,
            parallel_config=types.SimpleNamespace(pipeline_parallel_size=2, tensor_parallel_size=4),
            cache_config=object(),
            quant_config=object(),
            additional_config={"enable_cpu_binding": True},
        )
        self.subject = load_production_method()()
        self.subject.vllm_config = self.original_config
        self.subject.draft_model_config = types.SimpleNamespace(runner_type="draft", hf_config=object())
        self.subject.loaded_model = object()
        self.subject.loader_error = False

    def test_constructor_sees_draft_role_without_mutating_target(self):
        result = self.subject.load_draft_model("target", {"layer"})
        observed = self.subject.observed_config
        self.assertEqual(observed.model_config.runner_type, "draft")
        self.assertEqual(self.original_model_config.runner_type, "generate")
        self.assertIsNot(observed, self.original_config)
        self.assertIsNot(observed.model_config, self.original_model_config)
        self.assertIs(result, self.subject.loaded_model)

    def test_worker_topology_weights_and_cache_contracts_are_preserved(self):
        self.subject.load_draft_model("target", {"layer"})
        observed = self.subject.observed_config
        self.assertIs(observed.model_config.hf_config, self.original_model_config.hf_config)
        for key in ["parallel_config", "cache_config", "quant_config", "additional_config"]:
            self.assertIs(getattr(observed, key), getattr(self.original_config, key))
        self.assertEqual(observed.parallel_config.pipeline_parallel_size, 2)
        self.assertEqual(self.subject.observed_target, "target")
        self.assertEqual(self.subject.observed_names, {"layer"})

    def test_normal_return_restores_original_before_postprocessing(self):
        self.subject.load_draft_model("target", {"layer"})
        self.assertIs(self.subject.vllm_config, self.original_config)
        self.assertIs(self.subject.after_load_config, self.original_config)
        self.assertIs(self.subject.after_load_model, self.subject.loaded_model)

    def test_loader_exception_restores_original_config(self):
        self.subject.loader_error = True
        with self.assertRaisesRegex(RuntimeError, "loader failed"):
            self.subject.load_draft_model("target", {"layer"})
        self.assertIs(self.subject.vllm_config, self.original_config)
        self.assertEqual(self.original_model_config.runner_type, "generate")
        self.assertFalse(hasattr(self.subject, "after_load_model"))

    def test_repeated_loads_do_not_leak_or_nest_config(self):
        self.subject.load_draft_model("first", {"a"})
        first = self.subject.observed_config
        self.subject.load_draft_model("second", {"b"})
        second = self.subject.observed_config
        self.assertIsNot(first, second)
        self.assertIs(self.subject.vllm_config, self.original_config)
        self.assertEqual(second.model_config.runner_type, "draft")
        self.assertIs(second.model_config.hf_config, self.original_model_config.hf_config)
