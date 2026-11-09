# SPDX-License-Identifier: MIT
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bc250"))
from capacity_plan_d41 import (
    admit,
    headroom,
    parameters,
    CONFIG,
    PARAMETERS,
    GIB,
    HARD_MEMORY,
    SOFT_MEMORY,
    DISK_RESERVE,
    METADATA_RESERVE,
)


class CapacityTest(unittest.TestCase):
    def test_exact_shape_and_conservative_CPU_budget(self):
        self.assertEqual(parameters(CONFIG), PARAMETERS)
        plan = admit("large", 14 * GIB, 20 * GIB)
        self.assertEqual(plan["required_payload_bytes"], 12293652480)
        self.assertEqual(plan["largest_FP64_tensor_bytes"], 128 * 1024**2)
        self.assertLess(plan["planned_memory_bytes"], SOFT_MEMORY)
        self.assertEqual(HARD_MEMORY - SOFT_MEMORY, GIB)
        self.assertFalse(plan["GPU_admitted"])

    def test_disk_payload_reserve_admission_boundary(self):
        free = PARAMETERS * 12 + DISK_RESERVE + METADATA_RESERVE
        admit("large", 14 * GIB, free)
        with self.assertRaises(ValueError):
            admit("large", 14 * GIB, free - 1)

    def test_runtime_and_host_allowance_required(self):
        with self.assertRaises(ValueError):
            admit("large", 12 * GIB, 20 * GIB)
        with self.assertRaises(ValueError):
            admit("control", 12 * GIB - 1, 20 * GIB)
        with self.assertRaises(ValueError):
            admit("unknown", 14 * GIB, 20 * GIB)

    def test_soft_abort_precedes_hard_limit(self):
        headroom(SOFT_MEMORY - 1, 3 * GIB, DISK_RESERVE)
        for memory, available, free in [
            (SOFT_MEMORY, 3 * GIB, DISK_RESERVE),
            (SOFT_MEMORY - 1, 3 * GIB - 1, DISK_RESERVE),
            (SOFT_MEMORY - 1, 3 * GIB, DISK_RESERVE - 1),
        ]:
            with self.assertRaises(MemoryError):
                headroom(memory, available, free)


if __name__ == "__main__":
    unittest.main()
