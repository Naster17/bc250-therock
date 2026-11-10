# SPDX-License-Identifier: MIT
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bc250"))
import qualification_d43 as subject


class ProofTest(unittest.TestCase):
    def test_both_shapes_and_ranks_have_original_all_value_proofs(self):
        for shape in ["control", "large"]:
            for rank in range(2):
                item = subject.proof(shape, rank)
                self.assertEqual(item["worker"]["physical_rank"], rank)
                self.assertTrue(
                    item["worker"]["reference"]["ALL_AdamW_parameters_moments_counters"]
                )

    def test_descriptor_tampering_refused(self):
        with patch.object(subject, "digest", return_value="0" * 64):
            with self.assertRaisesRegex(ValueError, "descriptor changed"):
                subject.proof("large", 0)

    def test_model_source_change_refused_before_gpu(self):
        digest = subject.digest
        with patch.object(
            subject,
            "digest",
            side_effect=lambda p: "0" * 64 if p.name == "fixture_d35.py" else digest(p),
        ):
            with self.assertRaisesRegex(ValueError, "source changed"):
                subject.proof("large", 1)

    def test_unknown_recipe_or_physical_rank_refused(self):
        for shape, rank in [("single", 0), ("large", 2), ("control", -1)]:
            with self.assertRaises(ValueError):
                subject.proof(shape, rank)


if __name__ == "__main__":
    unittest.main()
