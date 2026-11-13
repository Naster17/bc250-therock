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


profile = load("profile_gated_delta_d68")
qualify = load("qualify_gated_delta_d68")


class ReportTest(unittest.TestCase):
    def test_structure(self):
        report = profile.new_report()
        self.assertEqual(report["fixture"], "profile_gated_delta_d68")
        self.assertEqual(report["status"], "running")
        self.assertEqual(len(report["geometries"]), 4)
        labels = [g["label"] for g in report["geometries"]]
        self.assertEqual(
            labels, ["a06-long", "a06-short", "d66-benchmark", "d66-long256"]
        )
        self.assertEqual(report["protocol"]["warmup"], 3)
        self.assertEqual(report["protocol"]["rounds"], 6)

    def test_geometry_chunk_counts(self):
        report = profile.new_report()
        by_label = {g["label"]: g for g in report["geometries"]}
        self.assertEqual(profile.chunk_count(by_label["a06-long"]["seq"]), 16)
        self.assertEqual(profile.chunk_count(by_label["a06-short"]["seq"]), 2)
        self.assertEqual(profile.chunk_count(by_label["d66-benchmark"]["seq"]), 1)
        self.assertEqual(profile.chunk_count(by_label["d66-long256"]["seq"]), 4)


class SummarizeTest(unittest.TestCase):
    def test_median_and_iqr(self):
        result = profile.summarize([4.0, 1.0, 2.0, 3.0, 5.0, 6.0])
        self.assertAlmostEqual(result["median_ms"], 3.5)
        # q1 = median([1,2,3]) = 2, q3 = median([4,5,6]) = 5
        self.assertAlmostEqual(result["iqr_percent"], (5.0 - 2.0) / 3.5 * 100.0)

    def test_single_sample(self):
        result = profile.summarize([7.5])
        self.assertEqual(result["median_ms"], 7.5)
        self.assertEqual(result["iqr_percent"], 0.0)

    def test_zero_median_no_division(self):
        result = profile.summarize([0.0, 0.0])
        self.assertEqual(result["iqr_percent"], 0.0)


class FractionTest(unittest.TestCase):
    def test_fraction(self):
        self.assertAlmostEqual(profile.fraction(2.0, 8.0), 0.25)

    def test_zero_whole(self):
        self.assertEqual(profile.fraction(2.0, 0.0), 0.0)


class ChunkCountTest(unittest.TestCase):
    def test_exact_and_partial(self):
        self.assertEqual(profile.chunk_count(64), 1)
        self.assertEqual(profile.chunk_count(1), 1)
        self.assertEqual(profile.chunk_count(65), 2)
        self.assertEqual(profile.chunk_count(1024), 16)


class QualifyReportTest(unittest.TestCase):
    def test_predeclared_bounds(self):
        report = qualify.new_report()
        bounds = report["parity_bounds"]
        self.assertEqual(bounds["op_rel_l2"]["float32"], 1e-5)
        self.assertEqual(bounds["op_rel_l2"]["bfloat16"], 1e-2)
        self.assertEqual(bounds["op_rel_l2"]["float16"], 1e-2)
        self.assertEqual(bounds["cpu_recurrence_tol"]["float32"], (2e-5, 2e-4))
        self.assertEqual(bounds["model_fp32_grad_rel_l2"], 2e-3)
        self.assertEqual(bounds["cpu_anchor_grad_rel_l2"], 5e-3)
        self.assertEqual(bounds["model_low_grad_rel_l2"], 0.04)

    def test_speedup(self):
        self.assertAlmostEqual(qualify.speedup(10.0, 2.5), 4.0)
        self.assertEqual(qualify.speedup(10.0, 0.0), 0.0)

    def test_summarize_single_sample(self):
        result = qualify.summarize([3.0])
        self.assertEqual(result["median_ms"], 3.0)
        self.assertEqual(result["iqr_percent"], 0.0)

    def test_load_candidate(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.py"
            path.write_text("VALUE = 68\n")
            module = qualify.load_candidate(str(path))
            self.assertEqual(module.VALUE, 68)


if __name__ == "__main__":
    unittest.main()
