# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import importlib.util
from pathlib import Path
import tempfile
import json
import os
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
                self.assertEqual(args, [str(script), "--timed"])
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
                    str(script),
                    "input.json",
                ],
            )
            self.assertEqual(mounts, [(script.parent, str(script.parent))])

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


class LauncherWorkflowsTest(unittest.TestCase):
    def test_launcher_options_stop_at_python_or_script_arguments(self):
        options = launcher.parse_launcher_arguments(
            ["--shm-size", "2g", "-u", "train.py", "--mount", "model-output"], {}
        )
        self.assertEqual(options.shm_size, "2g")
        self.assertEqual(
            options.arguments, ["-u", "train.py", "--mount", "model-output"]
        )
        options = launcher.parse_launcher_arguments(["--", "-m", "package"], {})
        self.assertEqual(options.arguments, ["-m", "package"])
        options = launcher.parse_launcher_arguments(["--shell", "-lc", "pwd"], {})
        self.assertTrue(options.shell)
        self.assertEqual(options.arguments, ["-lc", "pwd"])

    def test_environment_is_selected_and_values_are_not_in_command_arguments(self):
        options = launcher.parse_launcher_arguments(
            ["--env", "CUSTOM", "--env", "HF_HOME=/data/cache", "-c", "pass"],
            {
                "CUSTOM": "private-value",
                "HF_TOKEN": "private-token",
                "UNRELATED": "ignored",
            },
        )
        self.assertEqual(options.environment["CUSTOM"], "private-value")
        self.assertNotIn("UNRELATED", options.environment)
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            manifest = {"uid": os.getuid(), "gid": os.getgid(), "image": "test-image"}
            command = launcher.build_command(
                home / "release", manifest, home, home, options, [27, 27], home / "env"
            )
            self.assertNotIn("private-value", " ".join(command))
            self.assertNotIn("private-token", " ".join(command))
            self.assertIn(f"HOME={home}", command)
            self.assertEqual(command.count("27"), 1)
            self.assertEqual(command[command.index("-w") + 1], str(home))

    def test_environment_files_and_invalid_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.env"
            path.write_text("# comment\n\nCUSTOM=hello world=42\nHF_HOME=/data/cache\n")
            options = launcher.parse_launcher_arguments(["--env-file", str(path)], {})
            self.assertEqual(options.environment["CUSTOM"], "hello world=42")
        for arguments in (
            ["--env", "HOME=/root"],
            ["--env", "HSA_ENABLE_SDMA=1"],
            ["--env", "MISSING"],
            ["--env", "INVALID-NAME=x"],
            ["--env", "TOKEN=first\nsecond"],
            ["--shm-size", "0"],
            ["--shm-size", "1g;bad"],
            ["--mount"],
            ["--mount", "/"],
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                launcher.parse_launcher_arguments(arguments, {})

    def test_read_only_file_mount_overrides_automatic_mount(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            data = home / "data file.txt"
            data.touch()
            options = launcher.parse_launcher_arguments(
                ["--mount-ro", str(data), "-c", "pass"], {}
            )
            manifest = {"uid": os.getuid(), "gid": os.getgid(), "image": "test-image"}
            command = launcher.build_command(
                home / "release", manifest, home, home, options, [], home / "env"
            )
            self.assertIn(f"{data}:{data}:ro", command)
            self.assertNotIn(f"{home}:/workspace/project:rw", command)

    def test_symlinked_script_mounts_lexical_and_resolved_parents(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            target = home / "target"
            target.mkdir()
            script = target / "train.py"
            script.touch()
            alias = home / "alias"
            alias.mkdir()
            (alias / "train.py").symlink_to(script)
            cwd = home / "project"
            cwd.mkdir()
            args, mounts = launcher.prepare_python_arguments(
                [str(alias / "train.py")], cwd
            )
            self.assertEqual(args, [str(alias / "train.py")])
            self.assertEqual(set(mounts), {(alias, str(alias)), (target, str(target))})

    def test_read_only_parent_is_not_bypassed_by_automatic_cwd_mount(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            cwd = home / "project"
            cwd.mkdir()
            options = launcher.parse_launcher_arguments(
                ["--mount-ro", str(home), "-c", "pass"], {}
            )
            manifest = {"uid": os.getuid(), "gid": os.getgid(), "image": "test-image"}
            command = launcher.build_command(
                home / "release", manifest, cwd, home, options, [], home / "env"
            )
            self.assertIn(f"{cwd}:{cwd}:ro", command)
            self.assertNotIn(f"{cwd}:{cwd}:rw", command)

    def test_private_context_permissions_and_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "context.json"
            path.write_text(
                json.dumps(
                    {"options": {"arguments": ["-c", "pass"]}, "workspace": directory}
                )
            )
            path.chmod(0o600)
            options, cwd = launcher.read_context(path, os.getuid())
            self.assertEqual(options.arguments, ["-c", "pass"])
            self.assertEqual(cwd, Path(directory))
            path.chmod(0o644)
            with self.assertRaises(ValueError):
                launcher.read_context(path, os.getuid())
            alias = path.with_name("alias")
            alias.symlink_to(path)
            with self.assertRaises(OSError):
                launcher.read_context(alias, os.getuid())


if __name__ == "__main__":
    unittest.main()
