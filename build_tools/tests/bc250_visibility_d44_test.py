# SPDX-License-Identifier: MIT
"""Host-side regression for the specific retained D12 import crash input."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "bc250/bc250_torch.py"
spec = importlib.util.spec_from_file_location("visibility_launcher", SCRIPT)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class VisibilityTest(unittest.TestCase):
    def test_caller_and_explicit_override_refused(self):
        for arguments, environment in [
            (["-c", "pass"], {"HIP_VISIBLE_DEVICES": "-1"}),
            (["--env", "HIP_VISIBLE_DEVICES=-1", "-c", "pass"], {}),
            (
                ["--env", "HIP_VISIBLE_DEVICES", "-c", "pass"],
                {"HIP_VISIBLE_DEVICES": "-1"},
            ),
        ]:
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(ValueError, "recorded ROCr import crash"):
                    launcher.parse_launcher_arguments(arguments, environment)

    def test_file_order_and_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "env"
            path.write_text("HIP_VISIBLE_DEVICES=-1\n")
            with self.assertRaisesRegex(ValueError, "HIP_VISIBLE_DEVICES=-1"):
                launcher.parse_launcher_arguments(
                    ["--env-file", str(path), "-c", "pass"], {}
                )
            options = launcher.parse_launcher_arguments(
                [
                    "--env-file",
                    str(path),
                    "--env",
                    "HIP_VISIBLE_DEVICES=0",
                    "-c",
                    "pass",
                ],
                {},
            )
            self.assertEqual(options.environment["HIP_VISIBLE_DEVICES"], "0")

    def test_private_context_revalidated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "context.json"
            path.write_text(
                json.dumps(
                    dict(
                        workspace=directory,
                        options=vars(
                            launcher.LaunchOptions(
                                environment={"HIP_VISIBLE_DEVICES": "-1"}
                            )
                        ),
                    )
                )
            )
            path.chmod(0o600)
            with self.assertRaisesRegex(ValueError, "HIP_VISIBLE_DEVICES=-1"):
                launcher.read_context(path, os.getuid())

    def test_other_filter_semantics_preserved(self):
        # Accepted by the frontend only; this is not GPU qualification of filters.
        for name, values in {
            "HIP_VISIBLE_DEVICES": ["", "0", "GPU-0123456789abcdef", "0,-1"],
            "CUDA_VISIBLE_DEVICES": ["", "-1", "0"],
            "ROCR_VISIBLE_DEVICES": ["", "-1", "0", "GPU-0123456789abcdef"],
        }.items():
            for value in values:
                with self.subTest(name=name, value=value):
                    options = launcher.parse_launcher_arguments(
                        ["-c", "pass"], {name: value}
                    )
                    self.assertEqual(options.environment[name], value)


if __name__ == "__main__":
    unittest.main()
