# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

BC250 = Path(__file__).resolve().parents[1] / "bc250"


def load(name):
    spec = importlib.util.spec_from_file_location(name, BC250 / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


routing = load("route_qwen35_gated_delta_d69")
fixture = load("qualify_route_d69")
wrapper = load("d69_train_wrapper")


class GateTest(unittest.TestCase):
    def test_version_gate(self):
        self.assertTrue(routing.version_gate("5.22.1"))
        self.assertFalse(routing.version_gate("5.2.0"))
        self.assertFalse(routing.version_gate("5.22.0"))

    def test_model_type_gate(self):
        self.assertTrue(routing.model_type_gate("qwen3_5_text"))
        self.assertFalse(routing.model_type_gate("qwen3_5"))
        self.assertFalse(routing.model_type_gate(None))

    def test_kill_switch(self):
        self.assertFalse(routing.kill_switch_enabled({}))
        self.assertFalse(routing.kill_switch_enabled({"BC250_NATIVE_GATED_DELTA": "1"}))
        self.assertTrue(routing.kill_switch_enabled({"BC250_NATIVE_GATED_DELTA": "0"}))
        with self.assertRaises(ValueError):
            routing.kill_switch_enabled({"BC250_NATIVE_GATED_DELTA": "yes"})

    def test_geometry(self):
        self.assertTrue(routing.geometry_supported(1, 1024, 16, 128, 128))
        self.assertTrue(routing.geometry_supported(1, 8192, 16, 128, 128))
        self.assertFalse(routing.geometry_supported(1, 65, 2, 256, 9))
        self.assertFalse(routing.geometry_supported(1, 65, 2, 7, 9, chunk_size=32))
        self.assertFalse(routing.geometry_supported(0, 65, 2, 7, 9))
        # padded-token cap: 2 * 16 * 8192 = 262144 ok; beyond 1M refused
        self.assertTrue(routing.geometry_supported(2, 8192, 16, 128, 128))
        self.assertFalse(routing.geometry_supported(16, 8192, 16, 128, 128))

    def test_native_target_gate(self):
        self.assertTrue(routing.native_target_gate("installed"))
        self.assertTrue(routing.native_target_gate("fused"))
        self.assertFalse(routing.native_target_gate("bogus"))

    def test_classify_extra_kwargs(self):
        self.assertEqual(routing.classify_extra_kwargs({}), (None, []))
        self.assertEqual(
            routing.classify_extra_kwargs({"cu_seqlens": None}), (None, [])
        )
        self.assertEqual(
            routing.classify_extra_kwargs(
                {
                    "cu_seqlens": None,
                    "num_items_in_batch": 4,
                    "output_attentions": False,
                    "use_cache": False,
                }
            ),
            (None, ["num_items_in_batch", "output_attentions", "use_cache"]),
        )
        self.assertEqual(
            routing.classify_extra_kwargs({"cu_seqlens": "packed"})[0], "cu-seqlens"
        )
        self.assertEqual(
            routing.classify_extra_kwargs({"seq_idx": "x"})[0], "packed-kwargs"
        )
        self.assertEqual(
            routing.classify_extra_kwargs({"position_ids": "x"})[0], "packed-kwargs"
        )
        self.assertEqual(
            routing.classify_extra_kwargs({"is_causal": False})[0], "non-causal"
        )
        self.assertEqual(routing.classify_extra_kwargs({"is_causal": True}), (None, []))
        self.assertEqual(
            routing.classify_extra_kwargs({"mystery_kwarg": 1})[0], "kwargs"
        )

    def decision(self, **overrides):
        base = dict(
            query_dtype="float32",
            key_dtype="float32",
            value_dtype="float32",
            batch=2,
            seq=65,
            heads=2,
            key_width=7,
            value_width=9,
            chunk_size=64,
            initial_state_is_none=True,
            compiling=False,
        )
        base.update(overrides)
        return routing.routing_decision(**base)

    def test_routing_decision(self):
        self.assertEqual(self.decision(), "native")
        self.assertEqual(self.decision(compiling=True), "compiled-context")
        self.assertEqual(self.decision(extra_kwargs=True), "kwargs")
        self.assertEqual(self.decision(query_dtype="int32"), "dtype")
        self.assertEqual(self.decision(key_dtype="float16"), "dtype")
        self.assertEqual(self.decision(key_width=256), "geometry")
        self.assertEqual(self.decision(initial_state_is_none=False), "initial-state")


class AppModelTypeTest(unittest.TestCase):
    def test_nested_and_top_level(self):
        with tempfile.TemporaryDirectory() as tmp:
            nested = Path(tmp) / "nested"
            nested.mkdir()
            (nested / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3_5",
                        "text_config": {"model_type": "qwen3_5_text"},
                    }
                )
            )
            self.assertEqual(
                routing.app_model_type(nested / "config.json"), "qwen3_5_text"
            )
            top = Path(tmp) / "top"
            top.mkdir()
            (top / "config.json").write_text(json.dumps({"model_type": "qwen3_5_text"}))
            self.assertEqual(
                routing.app_model_type(top / "config.json"), "qwen3_5_text"
            )


class WrapperTest(unittest.TestCase):
    def test_extract_config(self):
        self.assertEqual(
            wrapper.extract_config(["--config", "/a/b.json", "--x"]),
            Path("/a/b.json"),
        )
        self.assertEqual(
            wrapper.extract_config(["--config=/a/b.json"]), Path("/a/b.json")
        )
        with self.assertRaises(ValueError):
            wrapper.extract_config(["--other", "1"])


class FixtureReportTest(unittest.TestCase):
    def test_predeclared_bounds(self):
        report = fixture.new_report()
        bounds = report["parity_bounds"]
        self.assertEqual(bounds["op_rel_l2"]["float32"], 1e-5)
        self.assertEqual(bounds["op_rel_l2"]["bfloat16"], 1e-2)
        self.assertEqual(bounds["model_fp32_grad_rel_l2"], 2e-3)
        self.assertEqual(bounds["cpu_anchor_grad_rel_l2"], 5e-3)
        self.assertEqual(len(report["geometries"]), 4)

    def test_tiny_config_overrides(self):
        overrides = fixture.tiny_config_overrides()
        self.assertEqual(overrides["num_hidden_layers"], 4)
        self.assertEqual(overrides["linear_num_value_heads"], 2)
        self.assertEqual(overrides["linear_key_head_dim"], 16)
        self.assertTrue(overrides["tie_word_embeddings"])


def _row(step, seconds, tokens=4096, skipped=False, finite=True):
    return {
        "step": step,
        "seconds": seconds,
        "optimizer_skipped": skipped,
        "finite_gradients": finite,
        "input_tokens": tokens,
        "peak_allocated_bytes": 1000,
    }


class SummarizeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "summarize_d69", BC250 / "route_port_d69" / "summarize_d69.py"
        )
        cls.summarize = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.summarize)

    def test_steady_excludes_first(self):
        rows = [_row(1, 50.0)] + [_row(i, 30.0 + i) for i in range(2, 7)]
        result = self.summarize.steady_state(rows)
        self.assertEqual(result["updates_run"], 6)
        self.assertEqual(result["updates_measured"], 5)
        self.assertAlmostEqual(result["median_s"], 34.0)
        self.assertEqual(result["tokens_per_update"], 4096)

    def test_steady_refuses_skipped(self):
        rows = [_row(1, 50.0), _row(2, 30.0, skipped=True)]
        with self.assertRaises(ValueError):
            self.summarize.steady_state(rows)

    def test_steady_refuses_token_disagreement(self):
        rows = [_row(1, 50.0), _row(2, 30.0), _row(3, 31.0, tokens=2048)]
        with self.assertRaises(ValueError):
            self.summarize.steady_state(rows)

    def test_cold_first(self):
        result = self.summarize.cold_first([_row(1, 326.8)])
        self.assertAlmostEqual(result["first_update_s"], 326.8)
        self.assertAlmostEqual(result["tokens_per_second"], 4096 / 326.8)

    def test_build_summary_ratios_and_stopped(self):
        cases = {
            "1024-ref": {"median_s": 36.0},
            "1024-fused": {"median_s": 18.0},
            "1024-installed": {"median_s": 35.0},
            "4096-ref": {"first_update_s": 320.0},
            "4096-fused": {"first_update_s": 160.0},
            "8192-fused": {"status": "stopped"},
        }
        summary = self.summarize.build_summary(cases)
        self.assertAlmostEqual(summary["ratios"]["1024-fused-vs-ref"], 2.0)
        self.assertAlmostEqual(summary["ratios"]["1024-installed-vs-ref"], 36 / 35)
        self.assertAlmostEqual(summary["ratios"]["4096-fused-vs-ref"], 2.0)
        self.assertNotIn("8192-fused-vs-ref", summary["ratios"])


if __name__ == "__main__":
    unittest.main()
