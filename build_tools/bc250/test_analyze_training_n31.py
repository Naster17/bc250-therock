#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Offline trace contract tests; no ROCm import or GPU required."""
import unittest

from analyze_training_n31 import analyze


def marker(name, begin, duration, category="user_annotation"):
    return dict(name="n31/" + name, cat=category, ph="X", ts=begin, dur=duration)


def runtime(correlation, timestamp, thread):
    return dict(
        cat="cuda_runtime", ts=timestamp, tid=thread, args={"correlation": correlation}
    )


def kernel(correlation, timestamp, duration):
    return dict(
        name="test-kernel",
        cat="kernel",
        ph="X",
        ts=timestamp,
        dur=duration,
        args={"correlation": correlation, "External id": 7, "device": 0},
    )


class AnalyzeTrace(unittest.TestCase):
    def fixture(self):
        return [
            marker("forward", 0, 10),
            marker("backward", 10, 20),
            marker("backward", 1000, 999, "gpu_user_annotation"),
            dict(
                name="aten::mm",
                cat="cpu_op",
                args={
                    "External id": 7,
                    "Input Dims": [[3, 4], [4, 5]],
                    "Input type": ["float", "float"],
                },
            ),
            runtime(1, 1, 1),
            runtime(2, 12, 99),
            runtime(3, 15, 100),
            kernel(1, 100, 4),
            kernel(2, 106, 5),
            kernel(3, 108, 2),
        ]

    def test_annotations_worker_threads_overlap_and_shape(self):
        result = analyze(self.fixture())
        self.assertEqual(result["kernel_count"], 3)
        self.assertEqual(result["kernel_sum_us"], 11)
        self.assertEqual(result["kernel_device_union_us"], 9)
        self.assertEqual(result["stages"]["forward"]["gpu_us"], 4)
        self.assertEqual(result["stages"]["backward"]["gpu_us"], 7)
        self.assertEqual(result["excluded_gpu_annotations"], 1)
        self.assertEqual(result["top_launch_operations"][0]["name"], "aten::mm")
        self.assertEqual(result["top_mm_shapes"][0]["input_shapes"], [[3, 4], [4, 5]])

    def test_missing_correlation_refused(self):
        events = self.fixture()
        events[-1]["args"]["correlation"] = 999
        with self.assertRaises(KeyError):
            analyze(events)

    def test_ambiguous_stage_refused(self):
        events = self.fixture() + [marker("optimizer", 0, 30)]
        with self.assertRaises(AssertionError):
            analyze(events)

    def test_nonpositive_duration_refused(self):
        events = self.fixture()
        events[-1]["dur"] = 0
        with self.assertRaises(AssertionError):
            analyze(events)


if __name__ == "__main__":
    unittest.main()
