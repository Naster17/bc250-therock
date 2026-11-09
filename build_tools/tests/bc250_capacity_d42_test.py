# SPDX-License-Identifier: MIT
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bc250"))
from capacity_plan_d42 import (
    descriptor,
    admit,
    headroom,
    READINESS_SECONDS,
    CHILD_SECONDS,
    GUARD_SECONDS,
)
from capacity_plan_d41 import GIB, PARAMETERS


class CapacityTest(unittest.TestCase):
    def test_pinned_oracles_match_shapes_and_sources(self):
        self.assertEqual(descriptor("large")["manifest"]["parameters"], PARAMETERS)
        self.assertEqual(descriptor("control")["manifest"]["parameters"], 352957440)
        with self.assertRaises(ValueError):
            descriptor("unknown")

    def test_host_and_device_budgets_are_independent(self):
        plan = admit("large", 14 * GIB, 6 * GIB)
        self.assertEqual(plan["local_FP32_state_floor_bytes"], PARAMETERS * 8)
        self.assertGreater(plan["single_resident_FP32_AdamW_floor_bytes"], 15 * GIB)
        admit("large", 13 * GIB, 6 * GIB, device_free=10 * GIB)
        with self.assertRaises(ValueError):
            admit("large", 12 * GIB, 6 * GIB)
        with self.assertRaises(ValueError):
            admit("large", 13 * GIB, 6 * GIB, device_free=8 * GIB)

    def test_no_payload_allocation_but_disk_reserve_retained(self):
        free = 5 * GIB + 64 * 1024**2
        admit("large", 14 * GIB, free)
        with self.assertRaises(ValueError):
            admit("large", 14 * GIB, free - 1)

    def test_owned_stop_thresholds(self):
        headroom(9 * GIB - 1, 3 * GIB, 5 * GIB)
        for values in [
            (9 * GIB, 3 * GIB, 5 * GIB),
            (0, 3 * GIB - 1, 5 * GIB),
            (0, 3 * GIB, 5 * GIB - 1),
        ]:
            with self.assertRaises(MemoryError):
                headroom(*values)

    def test_readiness_covers_wire_lower_bound_and_guard_margin(self):
        lower_bound = 2 * PARAMETERS * 12 * 8 / 1e9
        self.assertGreater(lower_bound, 120)
        self.assertGreater(READINESS_SECONDS, lower_bound)
        self.assertGreater(CHILD_SECONDS, READINESS_SECONDS + 600)
        self.assertGreaterEqual(GUARD_SECONDS - CHILD_SECONDS, 300)


if __name__ == "__main__":
    unittest.main()
