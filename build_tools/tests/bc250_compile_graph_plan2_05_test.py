# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pure (no torch/GPU) tests for the PLAN2-05 compile/graph harness.

GPU parity runs on native gfx1013 under the guardian; these tests prove the
bounds transcription, fixture grid, orchestrator configs and key source-level
contracts (fullgraph, capturable whole-step, dropout refusal, fresh recapture).
"""

from pathlib import Path
import sys
import unittest

BC250 = Path(__file__).resolve().parents[1] / "bc250"
if str(BC250) not in sys.path:
    sys.path.insert(0, str(BC250))

from compile_graph_plan2_05 import fixtures  # noqa: E402
from compile_graph_plan2_05 import run_compile_graph  # noqa: E402

PKG = run_compile_graph.LOCAL_TOOLS


class BoundsTranscriptionTest(unittest.TestCase):
    def test_compile_bounds(self):
        self.assertEqual(
            fixtures.COMPILE_CPU_BOUND,
            {"float32": 5e-4, "float16": 6e-3, "bfloat16": 4e-2},
        )
        self.assertEqual(
            fixtures.COMPILE_EAGER_BOUND,
            {"float32": 1e-5, "float16": 2e-3, "bfloat16": 1e-2},
        )

    def test_graph_bounds(self):
        self.assertEqual(fixtures.GRAPH_FP32_BOUND, 1e-5)
        self.assertEqual(fixtures.graph_bound("float16"), 2e-3)
        self.assertEqual(fixtures.graph_bound("bfloat16"), 1e-2)
        self.assertEqual(fixtures.graph_bound("float32"), 1e-5)

    def test_constants(self):
        self.assertEqual(fixtures.GRAPH_UPDATES, 8)
        self.assertEqual(fixtures.RESUME_UPDATES, 16)
        self.assertEqual(fixtures.WARMUPS, 2)
        self.assertEqual(fixtures.SAMPLES, 5)


class FixtureGridTest(unittest.TestCase):
    def test_compile_parts(self):
        self.assertEqual(fixtures.compile_fixtures(1), ["attention", "lora"])
        self.assertEqual(fixtures.compile_fixtures(2), ["embed_ce", "dynamic"])

    def test_unknown_part_refused(self):
        with self.assertRaises(ValueError):
            fixtures.compile_fixtures(3)

    def test_dynamic_lengths_ascending(self):
        self.assertEqual(
            fixtures.DYNAMIC_LENGTHS, tuple(sorted(fixtures.DYNAMIC_LENGTHS))
        )
        self.assertGreaterEqual(len(fixtures.DYNAMIC_LENGTHS), 3)


class OrchestratorConfigTest(unittest.TestCase):
    def test_required_keys(self):
        for script in ("compile", "graph", "resume"):
            config = run_compile_graph.guardian_config("cg-x", script, 1)
            self.assertTrue(
                {"mode", "label", "argv", "rank", "evidence", "cwd", "audit"}
                <= set(config)
            )
            self.assertEqual(config["mode"], "single")
            self.assertEqual(config["rank"], 0)

    def test_compile_argv_part(self):
        config = run_compile_graph.guardian_config("cg-x", "compile", 2)
        self.assertEqual(config["argv"][-2:], ["--part", "2"])
        self.assertIn("qualify_compile.py", " ".join(config["argv"]))

    def test_resume_argv_state_dir_no_part(self):
        config = run_compile_graph.guardian_config("cg-x", "resume", 1)
        argv = config["argv"]
        self.assertIn("--state-dir", argv)
        self.assertNotIn("--part", argv)
        self.assertIn("qualify_graph_resume.py", " ".join(argv))

    def test_report_paths_distinct(self):
        paths = {
            run_compile_graph.report_path("compile", 1),
            run_compile_graph.report_path("compile", 2),
            run_compile_graph.report_path("graph", 1),
            run_compile_graph.report_path("resume", 1),
        }
        self.assertEqual(len(paths), 4)

    def test_package_files_exist(self):
        for name in run_compile_graph.PACKAGE_FILES:
            self.assertTrue((PKG / name).exists(), name)

    def test_reuses_qualified_guardian(self):
        source = (PKG / "run_compile_graph.py").read_text()
        for token in (
            "guard_nf4.py",
            "bounded_launch.py",
            "kernel_noise.py",
            "stock_clock_check",
            "idle_check",
        ):
            self.assertIn(token, source)


class SourceContractTest(unittest.TestCase):
    def test_compile_uses_fullgraph(self):
        source = (PKG / "qualify_compile.py").read_text()
        self.assertIn("fullgraph=True", source)
        self.assertIn("generated_kernel_count", source)

    def test_graph_step_refuses_dropout(self):
        source = (PKG / "graph_step.py").read_text()
        self.assertIn("GraphCaptureError", source)
        self.assertIn("dropout", source)
        self.assertIn("torch.cuda.graph", source)

    def test_whole_step_capturable(self):
        source = (PKG / "qualify_graph.py").read_text()
        self.assertIn("capturable=whole_step", source)
        self.assertIn("make_graphed_callables", source)

    def test_resume_fresh_subprocess(self):
        source = (PKG / "qualify_graph_resume.py").read_text()
        self.assertIn("subprocess.run", source)
        self.assertIn("--phase", source)
        self.assertIn("GraphedStep(step, warmup=3).capture()", source)


if __name__ == "__main__":
    unittest.main()
