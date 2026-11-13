# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import ast
import importlib.util
from pathlib import Path
import sys
import unittest

BC250 = Path(__file__).resolve().parents[1] / "bc250"
D72_DIR = BC250 / "gemm_gap_d72"
D71_DIR = BC250 / "gemm_efficiency_d71"


def load(name, directory=BC250):
    spec = importlib.util.spec_from_file_location(name, directory / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = load("bench_gemm_d71")
# probe_gap_d72 does `import bench_gemm_d71` at module level; resolve it to the
# already-loaded module so the test needs no GPU/torch and no sys.path pollution.
sys.modules["bench_gemm_d71"] = bench
probe = load("probe_gap_d72")
summarize = load("summarize_d72", D72_DIR)
runner = load("run_d72", D72_DIR)


def desc(shape, dtype="float16"):
    return {"shape": shape, "stride": [shape[1], 1], "dtype": dtype}


def mm_row(a_shape, b_shape, count, dtype="float16"):
    k, n = b_shape
    return {
        "op": "mm",
        "args": [desc(a_shape, dtype), desc(b_shape, dtype)],
        "count": count,
        "rocblas": {
            "batch_count": 1,
            "input_type": dtype,
            "K": k,
            "N": n,
            "M": a_shape[0],
        },
    }


def census_doc():
    return {
        "layouts": [
            {
                "op": "bmm",
                "args": [],
                "count": 9000,
                "rocblas": {
                    "batch_count": 16,
                    "input_type": "float16",
                    "K": 128,
                    "N": 64,
                },
            },
            mm_row([1024, 2048], [2048, 128], 1560),
            mm_row([1024, 6144], [6144, 2048], 456),
            mm_row([1024, 2048], [2048, 6144], 624),
            mm_row([1024, 2048], [2048, 2048], 504, dtype="float32"),
            {**mm_row([1024, 2048], [2048, 128], 99), "op": "addmm"},
        ]
    }


def profile_doc():
    # D71 per-shape in-model profile: the large projection dominates by device
    # time even though 2048x6144 == 6144x2048 geometrically.
    return {
        "gemm_ops": [
            {
                "op": "aten::mm",
                "input_shapes": [[1024, 6144], [6144, 2048]],
                "count": 114,
                "self_device_ms": 3987.4,
                "device_ms_per_call": 34.98,
            },
            {
                "op": "aten::mm",
                "input_shapes": [[1024, 2048], [2048, 6144]],
                "count": 156,
                "self_device_ms": 1366.4,
                "device_ms_per_call": 8.76,
            },
            {
                "op": "aten::mm",
                "input_shapes": [[653, 2048], [2048, 32768], [653, 32768]],
                "count": 14,
                "self_device_ms": 423.8,
                "device_ms_per_call": 30.27,
            },
        ]
    }


class ProbeTest(unittest.TestCase):
    def test_alt_impl_flag_constant(self):
        self.assertEqual(probe.FP16_ALT_IMPL, 0x4)

    def test_thresholds_above_d71_noise_band(self):
        self.assertGreater(probe.MEANINGFUL_RATIO, 1.05)
        self.assertGreater(probe.DOMINANT_RATIO, probe.MEANINGFUL_RATIO)

    def test_select_mm_layouts_filters_and_sorts(self):
        rows = probe.select_mm_layouts(census_doc())
        self.assertTrue(all(r["op"] == "mm" for r in rows))
        self.assertTrue(all(r["rocblas"]["input_type"] == "float16" for r in rows))
        self.assertTrue(all(r["rocblas"]["batch_count"] == 1 for r in rows))
        counts = [r["count"] for r in rows]
        self.assertEqual(counts, sorted(counts, reverse=True))
        self.assertEqual(counts[0], 1560)

    def test_select_mm_layouts_limit(self):
        self.assertEqual(len(probe.select_mm_layouts(census_doc(), limit=2)), 2)

    def test_dominant_profile_op_by_device_time(self):
        top = probe.dominant_profile_op(profile_doc())
        self.assertEqual(top["input_shapes"], [[1024, 6144], [6144, 2048]])

    def test_dominant_layout_is_large_projection(self):
        dom = probe.dominant_layout(census_doc(), profile_doc())
        self.assertEqual(dom["args"][0]["shape"], [1024, 6144])
        self.assertEqual(dom["args"][1]["shape"], [6144, 2048])

    def test_dominant_layout_missing_shape_raises(self):
        with self.assertRaises(ValueError):
            probe.dominant_layout({"layouts": []}, profile_doc())

    def test_read_clock_temp_best_effort(self):
        reading = probe.read_clock_temp()
        self.assertIn("freq_hz", reading)
        self.assertIn("temp_c", reading)

    def test_window_call_count_bounded(self):
        # ~8 ms/call -> ~250 calls per 2 s window; clamped at both ends so a
        # window is always a bounded synchronized batch (the r1 sustained bug
        # queued an unbounded async backlog and produced a single window).
        self.assertEqual(probe.window_call_count(8.0), 250)
        self.assertEqual(probe.window_call_count(0.001), 2000)
        self.assertEqual(probe.window_call_count(100.0), 50)
        self.assertEqual(probe.window_call_count(0.0), 2000)


def alt_impl_report(bf_ratio=4.0, best_ratio=4.0, differ=True):
    return {
        "status": "passed",
        "dominant": {
            "backward_forward_ratio": bf_ratio,
            "best_backward_forward_ratio": best_ratio,
            "solution_sets_differ": differ,
            "rocblas_layout": {"M": 2048, "N": 1024, "K": 6144},
            "args": [desc([1024, 6144]), desc([6144, 2048])],
        },
    }


def sustained_report(degradation=1.0, clock=True, temp=60.0):
    return {
        "status": "passed",
        "degradation_ratio": degradation,
        "clock_visible": clock,
        "temp_c_max": temp,
    }


def locality_report(cold_hot=1.0, fresh_hot=1.0):
    return {
        "status": "passed",
        "cold_hot_ratio": cold_hot,
        "fresh_hot_ratio": fresh_hot,
    }


class SummarizerTest(unittest.TestCase):
    def test_alt_impl_dominant(self):
        summary = summarize.build_summary(
            alt_impl_report(4.0), sustained_report(1.0), locality_report(1.0)
        )
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["primary_cause"], "alt_impl_flag")
        self.assertEqual(summary["attribution"], "alt_impl_flag_dominant")
        self.assertTrue(any("fp16_alt_impl" in f for f in summary["findings"]))

    def test_alt_impl_contributor_band(self):
        summary = summarize.build_summary(
            alt_impl_report(1.8), sustained_report(1.0), locality_report(1.0)
        )
        self.assertEqual(summary["attribution"], "alt_impl_flag_contributor")

    def test_solution_sets_differ_finding(self):
        summary = summarize.build_summary(
            alt_impl_report(4.0, differ=True), sustained_report(), locality_report()
        )
        self.assertTrue(
            any("different Tensile solution" in f for f in summary["findings"])
        )

    def test_thermal_throttling(self):
        summary = summarize.build_summary(
            alt_impl_report(1.0), sustained_report(3.0), locality_report(1.0)
        )
        self.assertEqual(summary["primary_cause"], "thermal_throttling")
        self.assertEqual(summary["attribution"], "thermal_throttling_dominant")

    def test_memory_locality(self):
        summary = summarize.build_summary(
            alt_impl_report(1.0), sustained_report(1.0), locality_report(cold_hot=3.0)
        )
        self.assertEqual(summary["primary_cause"], "memory_locality")

    def test_fresh_locality_used_when_larger(self):
        summary = summarize.build_summary(
            alt_impl_report(1.0), sustained_report(1.0), locality_report(1.1, 2.0)
        )
        self.assertEqual(summary["primary_cause"], "memory_locality")
        self.assertAlmostEqual(summary["candidates"]["memory_locality"], 2.0)

    def test_not_reproduced(self):
        summary = summarize.build_summary(
            alt_impl_report(1.0), sustained_report(1.0), locality_report(1.0)
        )
        self.assertEqual(summary["attribution"], "not_reproduced_by_isolated_fixtures")
        self.assertIsNone(summary["primary_cause"])

    def test_failed_input(self):
        bad = alt_impl_report(4.0)
        bad["status"] = "failed"
        summary = summarize.build_summary(bad, sustained_report(), locality_report())
        self.assertEqual(summary["status"], "failed")


class GuardianReuseTest(unittest.TestCase):
    def test_guardian_logic_identical_to_d71(self):
        def body(path):
            tree = ast.parse(path.read_text())
            return ast.dump(ast.Module(body=tree.body[1:], type_ignores=[]))

        self.assertEqual(body(D72_DIR / "guard_d72.py"), body(D71_DIR / "guard_d71.py"))

    def test_bounded_launch_hardened_backstop(self):
        d72 = (D72_DIR / "bounded_launch.py").read_text()
        self.assertIn('command[:0] = ["timeout", "430"]', d72)
        for envelope in (
            '"--cpus=4"',
            '"--cpuset-cpus=0-3"',
            '"--memory=10g"',
            '"--memory-swap=12g"',
            '"--pids-limit=512"',
        ):
            self.assertIn(envelope, d72)


class OrchestratorTest(unittest.TestCase):
    def test_modes(self):
        self.assertEqual(runner.MODES, ("alt-impl", "sustained", "locality"))

    def test_guardian_configs(self):
        for name in runner.MODES:
            cfg = runner.guardian_config(name)
            joined = " ".join(cfg["argv"])
            self.assertEqual(cfg["label"], f"d72-{name}")
            self.assertEqual(cfg["mode"], "single")
            self.assertEqual(cfg["rank"], 0)
            self.assertNotIn("audit", cfg)
            self.assertIn("probe_gap_d72.py", joined)
            self.assertIn(f"--mode {name}", joined)
            self.assertIn("--census", joined)
            self.assertIn("--profile", joined)
            self.assertIn("--output", joined)
            self.assertIn(f"d72-{name}", cfg["evidence"])

    def test_base_env(self):
        joined = " ".join(runner.base_env())
        self.assertIn("PYTHONPATH=", joined)
        self.assertIn("OMP_NUM_THREADS=4", joined)

    def test_session_timeout_hardened(self):
        self.assertEqual(runner.SESSION_TIMEOUT, 900)


if __name__ == "__main__":
    unittest.main()
