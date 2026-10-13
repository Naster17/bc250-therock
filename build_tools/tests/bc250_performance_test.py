# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import copy
import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "bc250/compare_performance.py"
spec = importlib.util.spec_from_file_location("bc250_performance", SCRIPT)
performance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(performance)


class PerformanceComparisonTest(unittest.TestCase):
    def setUp(self):
        self.new = {
            "status": "passed",
            "fixture_sha256": "fixture",
            "arch": "gfx1013",
            "kernel": "same-kernel",
            "wgp_count": 20,
            "cpu_affinity": list(range(16)),
            "torch_threads": 4,
            "sdma": "0",
            "architecture_override": None,
            "samples": 5,
            "sample_target_ms": 200,
            "warmup_ms": 400,
            "torch": "new",
            "hip": "new",
            "cases": [
                {
                    "name": "gemm",
                    "status": "passed",
                    "inputs_sha256": "same-bytes",
                    "wall_median_ms": 2,
                    "wall_iqr_percent": 1,
                    "first_invocation_ms": 10,
                    "peak_temperature_millicelsius": 75000,
                }
            ],
        }
        self.old = copy.deepcopy(self.new)
        self.old["cases"][0]["wall_median_ms"] = 4

    def test_matched_validated_inputs_produce_a_ratio(self):
        report = performance.compare(self.new, self.old)
        self.assertEqual(report["cases"][0]["new_speedup"], 2)
        self.assertFalse(report["cases"][0]["variable"])

    def test_same_seed_but_different_tensor_bytes_is_rejected(self):
        self.old["cases"][0]["inputs_sha256"] = "different-bytes"
        with self.assertRaisesRegex(ValueError, "Tensor bytes differ"):
            performance.compare(self.new, self.old)

    def test_failed_case_or_different_controls_is_rejected(self):
        original = copy.deepcopy(self.old)
        for field, value in (
            ("torch_threads", 8),
            ("kernel", "other"),
            ("warmup_ms", 0),
        ):
            with self.subTest(field=field):
                self.old = copy.deepcopy(original)
                self.old[field] = value
                with self.assertRaises(ValueError):
                    performance.compare(self.new, self.old)
        self.old = copy.deepcopy(original)
        self.old["cases"][0]["status"] = "failed"
        with self.assertRaises(ValueError):
            performance.compare(self.new, self.old)

    def test_variable_samples_are_flagged(self):
        self.old["cases"][0]["wall_iqr_percent"] = 60
        self.assertTrue(performance.compare(self.new, self.old)["cases"][0]["variable"])


if __name__ == "__main__":
    unittest.main()
