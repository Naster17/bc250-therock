# SPDX-License-Identifier: MIT
"""Regression checks for the D55 full-build recipe boundary."""

from pathlib import Path
import subprocess
import sys
import tarfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bc250"))
from run_full_build_d55_on_host import recipe_env_prefix

KIT = (
    Path(__file__).resolve().parents[3]
    / "kernel/artifacts/source-d52/torch-source-d52-r1.tar.gz"
)


def recipe(profile):
    with tarfile.open(KIT) as archive:
        member = f"torch-source-d52-r1/recipes/build-fresh-{profile}.sh"
        return archive.extractfile(member).read().decode()


class RecipeBoundaryTest(unittest.TestCase):
    def test_prefix_matches_wrapper_sed_and_keeps_native_exports(self):
        for profile, nccl in (("daily", "0"), ("rccl-single", "1")):
            text = recipe(profile)
            prefix = recipe_env_prefix(text)
            sed = subprocess.run(
                ["sed", "/^python -m pip install /,$d"],
                input=text,
                text=True,
                capture_output=True,
                check=True,
            ).stdout
            self.assertEqual(prefix, sed)
            self.assertNotIn("python -m pip install", prefix)
            self.assertIn("export PYTORCH_ROCM_ARCH=gfx1013", prefix)
            self.assertIn(f"export USE_NCCL={nccl}", prefix)
            self.assertIn("CMAKE_FRESH=1", prefix)

    def test_refuses_recipe_without_pip_boundary(self):
        with self.assertRaises(AssertionError):
            recipe_env_prefix("export USE_ROCM=1\n")


if __name__ == "__main__":
    unittest.main()
