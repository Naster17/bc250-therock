# SPDX-License-Identifier: MIT
"""D33 external qualification must not admit unproved or changed training math."""
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest

BASE = Path(__file__).parents[1] / "bc250"
spec = importlib.util.spec_from_file_location(
    "qualification_d33", BASE / "qualification_d33.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class QualificationTest(unittest.TestCase):
    def test_qualified_sources_and_all_modes(self):
        for rank in [0, 1]:
            for mode in ["single", "ddp", "fsdp"]:
                proof, reference = module.load_qualification(mode, rank)
                self.assertTrue(
                    reference["reference"]["ALL_AdamW_parameters_moments_counters"]
                )
                self.assertEqual(proof["physical_rank"], rank)

    def test_unknown_mode_or_rank_refused(self):
        for mode, rank in [("arbitrary", 0), ("single", 2)]:
            with self.assertRaises(ValueError):
                module.load_qualification(mode, rank)

    def test_corrupted_proof_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copy(BASE / "distributed_d33_qualification_0.json", root)
            with (root / "distributed_d33_qualification_0.json").open("ab") as stream:
                stream.write(b" ")
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                module.load_qualification("single", 0, root)

    def test_changed_training_source_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in [*module.SOURCES, "distributed_d33_qualification_0.json"]:
                shutil.copy(BASE / name, root)
            with (root / "distributed_d32_capacity.py").open("ab") as stream:
                stream.write(b"\n# changed\n")
            with self.assertRaisesRegex(ValueError, "training source changed"):
                module.load_qualification("single", 0, root)
