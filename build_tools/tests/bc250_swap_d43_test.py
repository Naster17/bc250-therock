# SPDX-License-Identifier: MIT
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bc250"))
from verify_distributed_d43_exports import swap_occupancy


class SwapTest(unittest.TestCase):
    def test_zero_baseline(self):
        self.assertEqual(
            swap_occupancy(
                [dict(host_swap_used_bytes=0, **{"memory.swap.current": 0})]
            ),
            0,
        )

    def test_preexisting_unchanged_occupancy_is_reported(self):
        trace = [
            dict(host_swap_used_bytes=2756608, **{"memory.swap.current": 0})
            for _ in range(3)
        ]
        self.assertEqual(swap_occupancy(trace), 2756608)

    def test_growth_is_not_hidden_by_baseline(self):
        trace = [
            dict(host_swap_used_bytes=v, **{"memory.swap.current": 0})
            for v in [2756608, 2760704]
        ]
        with self.assertRaises(AssertionError):
            swap_occupancy(trace)

    def test_container_swap_refused(self):
        with self.assertRaises(AssertionError):
            swap_occupancy(
                [dict(host_swap_used_bytes=2756608, **{"memory.swap.current": 4096})]
            )


if __name__ == "__main__":
    unittest.main()
