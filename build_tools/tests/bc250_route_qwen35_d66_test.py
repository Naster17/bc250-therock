# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import importlib.util
from pathlib import Path
import unittest

BC250 = Path(__file__).resolve().parents[1] / "bc250"


def load(name):
    spec = importlib.util.spec_from_file_location(name, BC250 / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


routing = load("route_qwen35_gated_delta_d66")
profile = load("profile_hybrid_d66")


class KillSwitchTest(unittest.TestCase):
    def test_default_allows_routing(self):
        self.assertFalse(routing.kill_switch_enabled({}))
        self.assertFalse(routing.kill_switch_enabled({routing.KILL_SWITCH: "1"}))

    def test_zero_refuses(self):
        self.assertTrue(routing.kill_switch_enabled({routing.KILL_SWITCH: "0"}))

    def test_invalid_value_raises(self):
        with self.assertRaises(ValueError):
            routing.kill_switch_enabled({routing.KILL_SWITCH: "2"})


class GateTest(unittest.TestCase):
    def test_version_gate(self):
        self.assertTrue(routing.version_gate("5.2.0"))
        self.assertFalse(routing.version_gate("5.19.0"))
        self.assertFalse(routing.version_gate("5.22.1"))

    def test_model_type_gate(self):
        self.assertTrue(routing.model_type_gate("qwen3_5_text"))
        self.assertFalse(routing.model_type_gate("qwen3"))
        self.assertFalse(routing.model_type_gate(None))

    def test_geometry_benchmark_case(self):
        self.assertTrue(routing.geometry_supported(2, 32, 2, 16, 16))

    def test_geometry_refusals(self):
        self.assertFalse(routing.geometry_supported(2, 32, 2, 256, 16))
        self.assertFalse(routing.geometry_supported(2, 32, 2, 16, 16, chunk_size=32))
        self.assertFalse(routing.geometry_supported(0, 32, 2, 16, 16))
        self.assertFalse(routing.geometry_supported(2, 0, 2, 16, 16))

    def test_geometry_padded_token_bound(self):
        self.assertTrue(routing.geometry_supported(128, 512, 16, 16, 16))
        self.assertFalse(routing.geometry_supported(129, 512, 16, 16, 16))

    def test_dtype_supported(self):
        self.assertTrue(routing.dtype_supported("float32", "float32", "float32"))
        self.assertTrue(routing.dtype_supported("bfloat16", "bfloat16", "bfloat16"))
        self.assertFalse(routing.dtype_supported("float32", "float16", "float32"))
        self.assertFalse(routing.dtype_supported("int8", "int8", "int8"))


class RoutingDecisionTest(unittest.TestCase):
    def base(self, **overrides):
        kwargs = dict(
            query_dtype="float32",
            key_dtype="float32",
            value_dtype="float32",
            batch=2,
            seq=32,
            heads=2,
            key_width=16,
            value_width=16,
            chunk_size=64,
            initial_state_is_none=True,
            compiling=False,
        )
        kwargs.update(overrides)
        return routing.routing_decision(**kwargs)

    def test_native(self):
        self.assertEqual(self.base(), "native")

    def test_compiled_context_falls_back(self):
        self.assertEqual(self.base(compiling=True), "compiled-context")

    def test_dtype_falls_back(self):
        self.assertEqual(self.base(key_dtype="float16"), "dtype")

    def test_geometry_falls_back(self):
        self.assertEqual(self.base(key_width=256), "geometry")

    def test_initial_state_falls_back(self):
        self.assertEqual(self.base(initial_state_is_none=False), "initial-state")


class ProfilePureLogicTest(unittest.TestCase):
    def test_report_schema(self):
        report = profile.new_report()
        self.assertEqual(report["fixture"], "profile_hybrid_d66")
        self.assertEqual(report["model"], "qwen35-hybrid-tiny")
        self.assertEqual(report["status"], "running")
        self.assertIn("op_rel_l2", report["parity_bounds"])
        self.assertEqual(report["parity_bounds"]["model_low_grad_rel_l2"], 0.04)

    def test_fp32_bounds_record_original_and_revision(self):
        bounds = profile.new_report()["parity_bounds"]
        self.assertEqual(bounds["model_fp32_grad_rel_l2_original"], 1e-4)
        self.assertEqual(bounds["model_fp32_grad_rel_l2"], 2e-3)
        self.assertEqual(bounds["model_fp32_logits_rel_l2"], 1e-4)
        self.assertEqual(bounds["cpu_anchor_grad_rel_l2"], 5e-3)
        self.assertLess(
            bounds["model_fp32_grad_rel_l2"], bounds["model_low_grad_rel_l2"]
        )

    def test_rel_l2_bounds(self):
        self.assertEqual(profile.rel_l2_bound("float32"), 1e-5)
        self.assertEqual(profile.rel_l2_bound("bfloat16"), 1e-2)
        self.assertEqual(profile.rel_l2_bound("float16"), 1e-2)

    def test_speedup(self):
        self.assertAlmostEqual(profile.speedup(35.742, 11.234), 35.742 / 11.234)

    def test_summarize_pairs(self):
        summary = profile.summarize_pairs([10.0, 12.0, 14.0, 16.0, 18.0])
        self.assertEqual(summary["median_ms"], 14.0)
        self.assertGreater(summary["iqr_percent"], 0.0)
        self.assertEqual(len(summary["samples_ms"]), 5)

    def test_sections_registry(self):
        self.assertEqual(
            set(profile.SECTIONS),
            {
                "breakdown",
                "op_timing",
                "model_parity",
                "admission_routed",
                "model_timing",
                "refusals",
            },
        )

    def test_workload_matches_d64(self):
        self.assertEqual(profile.BATCH, 2)
        self.assertEqual(profile.SEQ, 32)


if __name__ == "__main__":
    unittest.main()
