# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "bc250/bc250_torch.py"
spec = importlib.util.spec_from_file_location("bc250_launcher", SCRIPT)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherPathsTest(unittest.TestCase):
    def test_absolute_and_relative_scripts_inside_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            script = cwd / "bc250work/torch matrix.py"
            script.parent.mkdir()
            script.touch()
            for argument in (str(script), "bc250work/torch matrix.py"):
                args, mounts = launcher.prepare_python_arguments(
                    [argument, "--timed"], cwd
                )
                self.assertEqual(
                    args, ["/workspace/project/bc250work/torch matrix.py", "--timed"]
                )
                self.assertEqual(mounts, [])

    def test_external_script_with_python_options_and_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cwd = root / "project"
            cwd.mkdir()
            script = root / "scripts/model.py"
            script.parent.mkdir()
            script.touch()
            args, mounts = launcher.prepare_python_arguments(
                ["-u", "-X", "dev", "-W", "ignore", str(script), "input.json"], cwd
            )
            self.assertEqual(
                args,
                [
                    "-u",
                    "-X",
                    "dev",
                    "-W",
                    "ignore",
                    "/workspace/script/model.py",
                    "input.json",
                ],
            )
            self.assertEqual(mounts, [(script.parent, "/workspace/script")])

    def test_code_module_and_stdin_arguments_are_preserved(self):
        for args in (
            ["-c", "print('ok')"],
            ["-m", "pytest", "some_file.py"],
            ["-"],
            [],
        ):
            self.assertEqual(
                launcher.prepare_python_arguments(args, Path("/tmp")), (args, [])
            )

    def test_missing_script_fails_on_host(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                launcher.prepare_python_arguments(["missing.py"], Path(directory))


if __name__ == "__main__":
    unittest.main()
