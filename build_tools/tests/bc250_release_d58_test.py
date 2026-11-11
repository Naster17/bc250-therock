# SPDX-License-Identifier: MIT
"""Regression checks for the D58 release runtime-metadata adaptation."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bc250"))
from assemble_release_d58 import adapt_runtime


class AdaptRuntimeTest(unittest.TestCase):
    def test_release_fields_set_and_provenance_preserved(self):
        data = {
            "release": "dev-priority2",
            "validated": True,
            "mutable": True,
            "image": "1bb4ac0e",
            "torch_revision": "a524980",
            "quant_b052": {"keep": "history"},
            "rccl_single_profile": {"environment": {"NCCL_P2P_DISABLE": "1"}},
        }
        out = adapt_runtime(
            data, "10.2-gfx1013-20261111-138a66f52", "daily", {"task": "D58"}
        )
        self.assertEqual(out["release"], "10.2-gfx1013-20261111-138a66f52")
        self.assertFalse(out["validated"])
        self.assertFalse(out["mutable"])
        self.assertEqual(out["assembly"], {"task": "D58", "profile": "daily"})
        self.assertEqual(out["image"], "1bb4ac0e")
        self.assertEqual(out["torch_revision"], "a524980")
        self.assertEqual(out["quant_b052"], {"keep": "history"})
        self.assertEqual(out["rccl_single_profile"], data["rccl_single_profile"])
        self.assertEqual(data["release"], "dev-priority2")
        self.assertTrue(data["validated"])


if __name__ == "__main__":
    unittest.main()
