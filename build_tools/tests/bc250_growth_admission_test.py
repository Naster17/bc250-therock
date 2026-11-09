# SPDX-License-Identifier: MIT
"""D34 refuses unknown modes and insufficient shared-memory headroom."""
import importlib.util
from pathlib import Path
import unittest

BASE = Path(__file__).parents[1] / "bc250"
spec = importlib.util.spec_from_file_location(
    "capacity_plan_d34", BASE / "capacity_plan_d34.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class GrowthAdmissionTest(unittest.TestCase):
    def test_other_modes_refuse(self):
        for mode in ["ddp", "large", "", "FSDP"]:
            with self.assertRaises(ValueError):
                module.admit(mode, 20 * module.GIB, 20 * module.GIB)

    def test_insufficient_headroom_refuses_each_mode(self):
        for mode in ["single", "fsdp"]:
            for host, device in [
                (10 * module.GIB - 1, 20 * module.GIB),
                (20 * module.GIB, 6 * module.GIB - 1),
                (0, 0),
            ]:
                with self.assertRaises(ValueError):
                    module.admit(mode, host, device)

    def test_candidate_state_floor_and_position_length(self):
        config = module.CONFIG
        width = config["width"]
        expected = (
            config["blocks"] * (12 * width * width + 13 * width)
            + config["vocab"] * width
            + config["positions"] * width
            + 2 * width
        )
        self.assertEqual(expected, module.PARAMETERS)
        self.assertGreater(config["positions"], config["tokens"])
        for mode, divisor in [("single", 1), ("fsdp", 2)]:
            plan = module.admit(mode, 10 * module.GIB, 6 * module.GIB)
            self.assertEqual(plan["fp32_state_floor_bytes"], expected * 16 // divisor)
            self.assertFalse(plan["larger_than_single_fit"])
