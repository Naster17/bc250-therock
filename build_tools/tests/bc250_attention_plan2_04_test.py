# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pure (no torch/GPU) tests for the PLAN2-04 attention qualification harness.

The GPU parity/benchmark runs on native gfx1013 under the guardian; these tests
prove the fixture grid covers the pre-declared scope, the bounds transcription
matches BOUNDS.md, the refusal surface is complete, the forward-policy mirror is
correct and the orchestrator builds valid guardian configs.
"""

from pathlib import Path
import sys
import unittest

BC250 = Path(__file__).resolve().parents[1] / "bc250"
if str(BC250) not in sys.path:
    sys.path.insert(0, str(BC250))

from attention_plan2_04 import fixtures  # noqa: E402
from attention_plan2_04 import run_attention  # noqa: E402


class BoundsTranscriptionTest(unittest.TestCase):
    def test_rel_l2_bounds(self):
        self.assertEqual(
            fixtures.REL_L2_BOUNDS,
            {"float32": 3e-5, "float16": 5e-3, "bfloat16": 3e-2},
        )

    def test_close_bounds(self):
        self.assertEqual(
            fixtures.CLOSE_BOUNDS,
            {
                "float32": (5e-4, 3e-5),
                "float16": (8e-3, 3e-3),
                "bfloat16": (5e-2, 3e-2),
            },
        )

    def test_every_dtype_has_bounds(self):
        for dtype in fixtures.DTYPES:
            self.assertIn(dtype, fixtures.REL_L2_BOUNDS)
            self.assertIn(dtype, fixtures.CLOSE_BOUNDS)

    def test_timing_protocol_constants(self):
        self.assertEqual(fixtures.WARMUPS, 2)
        self.assertEqual(fixtures.SAMPLES, 5)
        self.assertEqual(fixtures.DRIFT_LIMIT, 1.25)


class ParityGridTest(unittest.TestCase):
    def test_geometries_valid(self):
        for part in (1, 2):
            for name, b, h, hk, m, n, d, mode in fixtures.parity_geometries(part):
                self.assertGreaterEqual(b, 1)
                self.assertGreaterEqual(h, 1)
                self.assertGreaterEqual(hk, 1)
                self.assertEqual(h % hk, 0, name)
                self.assertTrue(1 <= d <= 256, name)
                self.assertTrue(1 <= max(m, n) <= 4096, name)
                self.assertIn(mode, ("none", "causal", "bool", "additive"), name)

    def test_part1_covers_masks_cross_attention_head256(self):
        rows = fixtures.parity_geometries(1)
        modes = {r[7] for r in rows}
        self.assertEqual(modes, {"none", "causal", "bool", "additive"})
        self.assertTrue(any(r[4] != r[5] for r in rows))  # M != N
        self.assertTrue(any(r[6] == 256 for r in rows))
        self.assertTrue(any(r[2] != r[3] for r in rows))  # GQA
        self.assertTrue(all(max(r[4], r[5]) <= 512 for r in rows))

    def test_part2_covers_long_sequence_full_attention(self):
        rows = fixtures.parity_geometries(2)
        self.assertTrue(all(max(r[4], r[5]) >= 1024 for r in rows))
        self.assertTrue(any(r[4] == 4096 and r[7] == "none" for r in rows))
        self.assertTrue(any(r[4] == 4096 and r[7] == "causal" for r in rows))
        self.assertTrue(any(r[6] == 256 for r in rows))
        self.assertTrue(any(r[2] != r[3] for r in rows))

    def test_names_unique(self):
        names = [r[0] for r in fixtures.PARITY_PART1 + fixtures.PARITY_PART2]
        self.assertEqual(len(names), len(set(names)))

    def test_unknown_part_refused(self):
        with self.assertRaises(ValueError):
            fixtures.parity_geometries(3)


class BenchGridTest(unittest.TestCase):
    def test_part1_shape(self):
        rows = fixtures.bench_rows(1)
        self.assertEqual(len(rows), 32)
        self.assertEqual({r[3] for r in rows}, {64, 128})
        self.assertEqual({r[4] for r in rows}, {512, 1024, 2048, 4096})
        self.assertEqual({r[5] for r in rows}, {False, True})
        self.assertEqual({r[6] for r in rows}, {"float16", "bfloat16"})

    def test_part2_shape(self):
        rows = fixtures.bench_rows(2)
        self.assertEqual(len(rows), 9)
        self.assertTrue(any(r[3] == 256 and r[6] == "bfloat16" for r in rows))
        self.assertTrue(any(r[6] == "float32" and r[4] == 4096 for r in rows))
        self.assertTrue(any(r[3] == 256 and r[4] == 2048 for r in rows))

    def test_names_unique(self):
        names = [r[0] for r in fixtures.bench_rows(1) + fixtures.bench_rows(2)]
        self.assertEqual(len(names), len(set(names)))

    def test_unknown_part_refused(self):
        with self.assertRaises(ValueError):
            fixtures.bench_rows(0)


class PolicyMirrorTest(unittest.TestCase):
    def test_hybrid_only_bf16_head256_training(self):
        self.assertEqual(
            fixtures.policy_for("bfloat16", 256, True),
            "hybrid-math-forward-tiled-backward",
        )

    def test_pure_otherwise(self):
        for dtype, d, grad in (
            ("bfloat16", 256, False),
            ("bfloat16", 128, True),
            ("float16", 256, True),
            ("float32", 256, True),
            ("float16", 64, True),
        ):
            self.assertEqual(fixtures.policy_for(dtype, d, grad), "pure-tiled")


class RefusalSurfaceTest(unittest.TestCase):
    def test_complete_and_unique(self):
        names = [r[0] for r in fixtures.REFUSALS]
        self.assertEqual(len(names), 13)
        self.assertEqual(len(names), len(set(names)))
        expected = {
            "dropout",
            "mask-gradient",
            "causal-plus-mask",
            "gqa-without-flag",
            "nonfinite-scale",
            "head-dim-257",
            "seq-4097",
            "dtype-float64",
            "dtype-mismatch",
            "cpu-device",
            "wrong-rank",
            "higher-order-grad",
            "compile-fullgraph",
        }
        self.assertEqual(set(names), expected)

    def test_exception_names_valid(self):
        valid = {"NotImplementedError", "ValueError", "RuntimeError", "Exception"}
        for _, exc in fixtures.REFUSALS:
            self.assertIn(exc, valid)


class OrchestratorConfigTest(unittest.TestCase):
    def test_guardian_config_required_keys(self):
        config = run_attention.guardian_config("attn-r1", "qualify", 1)
        self.assertTrue(
            {"mode", "label", "argv", "rank", "evidence", "cwd"} <= set(config)
        )
        self.assertEqual(config["mode"], "single")
        self.assertEqual(config["rank"], 0)
        self.assertEqual(config["audit"], run_attention.report_path("qualify", 1))

    def test_argv_targets_requested_script_and_part(self):
        for script in ("qualify", "bench"):
            for part in (1, 2):
                config = run_attention.guardian_config("attn-x", script, part)
                argv = config["argv"]
                self.assertIn(f"{script}_attention.py", " ".join(argv))
                self.assertEqual(argv[-2:], ["--part", str(part)])
                self.assertIn(run_attention.report_path(script, part), argv)

    def test_report_paths_distinct(self):
        paths = {
            run_attention.report_path(s, p)
            for s in ("qualify", "bench")
            for p in (1, 2)
        }
        self.assertEqual(len(paths), 4)

    def test_package_files_exist(self):
        for name in run_attention.PACKAGE_FILES:
            self.assertTrue((run_attention.LOCAL_TOOLS / name).exists(), name)

    def test_reuses_qualified_guardian(self):
        source = (run_attention.LOCAL_TOOLS / "run_attention.py").read_text()
        self.assertIn("guard_nf4.py", source)
        self.assertIn("bounded_launch.py", source)
        self.assertIn("kernel_noise.py", source)
        self.assertIn("stock_clock_check", source)
        self.assertIn("idle_check", source)

    def test_bench_forwards_opts_to_every_timed_call(self):
        # Regression for the r4/r5 defect: timing helpers dropped is_causal/
        # enable_gqa, mislabeling causal fixtures and crashing GQA.
        source = (run_attention.LOCAL_TOOLS / "bench_attention.py").read_text()
        self.assertIn("def timed_call(fn, xs, dy, mode, graph_holder, opts)", source)
        self.assertIn("def time_pass(fn, xs, dy, mode, label, opts)", source)
        self.assertIn("def peak_extra(fn, xs, dy, opts)", source)
        self.assertIn("time_pass(fn, xs, dy, mode, label, opts)", source)
        self.assertIn("peak_extra(fn, xs, dy, opts)", source)
        self.assertNotIn("fn(*xs)\n", source)


if __name__ == "__main__":
    unittest.main()
