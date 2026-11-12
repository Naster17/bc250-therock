# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "bc250/benchmark_training_d64.py"
spec = importlib.util.spec_from_file_location("benchmark_training_d64", SCRIPT)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class CaseMatrixTest(unittest.TestCase):
    def test_covers_registry_backends_times_dtypes(self):
        cases = bench.case_matrix()
        backends = sum(
            len(s["backends"]) for s in bench.admission.MODEL_REGISTRY.values()
        )
        self.assertEqual(len(cases), backends * len(bench.DTYPES))

    def test_cases_are_well_formed(self):
        for name, dtype, backend in bench.case_matrix():
            self.assertIn(name, bench.admission.MODEL_REGISTRY)
            self.assertIn(dtype, bench.DTYPES)
            self.assertIn(backend, bench.admission.MODEL_REGISTRY[name]["backends"])

    def test_three_dtypes_present(self):
        self.assertEqual(set(bench.DTYPES), {"float32", "bfloat16", "float16"})


class MetricsTest(unittest.TestCase):
    def test_tokens_per_second(self):
        self.assertEqual(bench.tokens_per_second(128, 10.0), 12800.0)

    def test_iqr_percent(self):
        samples = [10.0, 12.0, 14.0, 16.0, 18.0]
        self.assertGreater(bench.iqr_percent(samples), 0.0)

    def test_summarize_case(self):
        summary = bench.summarize_case([10.0, 12.0, 14.0, 16.0, 18.0], 128)
        self.assertEqual(summary["step_median_ms"], 14.0)
        self.assertAlmostEqual(summary["tokens_per_second"], 128 * 1000.0 / 14.0)
        self.assertIn("step_iqr_percent", summary)
        self.assertEqual(len(summary["step_samples_ms"]), 5)


class ReportTest(unittest.TestCase):
    def test_new_report_schema(self):
        report = bench.new_report()
        self.assertEqual(report["fixture"], "benchmark_training_d64")
        self.assertEqual(report["workload"]["lr"], 0)
        self.assertEqual(report["workload"]["batch"], bench.BATCH)
        self.assertEqual(report["warmup_steps"], bench.WARMUP)
        self.assertEqual(report["intervals"], bench.INTERVALS)
        self.assertEqual(report["cases"], [])

    def test_read_utilization_returns_boundary_dict(self):
        info = bench.read_utilization()
        self.assertIn("gpu_busy_percent", info)
        self.assertIn("note", info)
        self.assertIsInstance(info, dict)


if __name__ == "__main__":
    unittest.main()
