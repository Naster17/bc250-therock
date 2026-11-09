# SPDX-License-Identifier: MIT
"""D32 refuses unsafe resource admission before any Torch/model import."""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).parents[1] / "bc250/capacity_plan_d32.py"
spec = importlib.util.spec_from_file_location("capacity_d32", path)
plan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan)


class AdmissionTest(unittest.TestCase):
    def test_host_and_device_headroom_are_both_required(self):
        for mode in ["single", "ddp", "fsdp"]:
            with self.subTest(mode=mode):
                self.assertFalse(
                    plan.admit(mode, 12 * plan.GIB, 6 * plan.GIB)["measured_larger_fit"]
                )
                for host, device in [
                    (9 * plan.GIB, 6 * plan.GIB),
                    (12 * plan.GIB, 3 * plan.GIB),
                ]:
                    with self.assertRaises(ValueError):
                        plan.admit(mode, host, device)

    def test_unknown_mode_cannot_bypass_admission(self):
        with self.assertRaises(ValueError):
            plan.admit("unbounded", 100 * plan.GIB, 100 * plan.GIB)
