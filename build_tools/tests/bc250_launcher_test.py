# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import importlib.util
from pathlib import Path
import tempfile
import json
import os
import unittest
import hashlib


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
    def test_profile_survives_context_and_stops_at_script_arguments(self):
        options = launcher.parse_launcher_arguments(
            ["--env", "CUSTOM=x", "--runtime-profile", "rccl-single", "-c", "pass"], {}
        )
        self.assertEqual(options.runtime_profile, "rccl-single")
        self.assertEqual(options.arguments, ["-c", "pass"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "context.json"
            path.write_text(
                json.dumps({"options": vars(options), "workspace": directory})
            )
            path.chmod(0o600)
            recovered, _ = launcher.read_context(path, os.getuid())
            self.assertEqual(recovered.runtime_profile, "rccl-single")
        with self.assertRaises(ValueError):
            launcher.parse_launcher_arguments(["--runtime-profile", "unknown"], {})
        default = launcher.parse_launcher_arguments(
            ["-c", "pass", "--runtime-profile", "rccl-single"], {}
        )
        self.assertEqual(default.runtime_profile, "daily")

    def test_rccl_profile_mount_settings_and_identity_refusals(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            release = home / "release"
            identities = {}
            for name in ("torch/torch/lib/libtorch_hip.so", "rccl/lib/librccl.so.1.0"):
                path = release / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
                identities[name] = hashlib.sha256(b"fixture").hexdigest()
            for name in ("librccl.so", "librccl.so.1"):
                (release / "rccl/lib" / name).symlink_to("librccl.so.1.0")
            manifest = dict(
                uid=os.getuid(),
                gid=os.getgid(),
                image="test-image",
                rccl_single_profile=dict(
                    environment=dict(launcher.RCCL_SINGLE_ENVIRONMENT),
                    identity_sha256=identities,
                ),
            )
            options = launcher.LaunchOptions(
                arguments=["-c", "pass"], runtime_profile="rccl-single"
            )
            command = launcher.build_command(
                release, manifest, home, home, options, [], home / "env"
            )
            self.assertIn(f"{release / 'rccl'}:/workspace/rccl:ro", command)
            self.assertIn(
                "LD_LIBRARY_PATH=/workspace/rccl/lib:/workspace/build/dist/rocm/lib:/workspace/build/dist/rocm/lib/llvm/lib",
                command,
            )
            for name, value in launcher.RCCL_SINGLE_ENVIRONMENT.items():
                self.assertIn(f"{name}={value}", command)
                options.environment = {name: "unsafe"}
                with self.assertRaises(ValueError):
                    launcher.qualified_rccl_profile(release, manifest, options)
            options.environment = {}
            options.runtime_profile = "daily"
            with self.assertRaises(ValueError):
                launcher.qualified_rccl_profile(release, manifest, options)
            options.runtime_profile = "rccl-single"
            (release / "rccl/lib/librccl.so.1.0").write_bytes(b"mismatch")
            with self.assertRaises(ValueError):
                launcher.qualified_rccl_profile(release, manifest, options)

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


class NegativeVisibilityAllowlistTest(unittest.TestCase):
    def _release(self, directory, runtime_text, sdk_bytes, aliases=True):
        release = Path(directory)
        (release / "runtime.json").write_text(runtime_text)
        lib = release / "rocm/lib"
        lib.mkdir(parents=True)
        target = lib / "librocprofiler-sdk.so.1.4.2"
        target.write_bytes(sdk_bytes)
        if aliases:
            for name in ("librocprofiler-sdk.so", "librocprofiler-sdk.so.1"):
                (lib / name).symlink_to(target.name)
        return release

    def test_bound_cohorts_admitted_and_unknown_refused(self):
        sdk = b"qualified-sdk"
        sdk_sha = hashlib.sha256(sdk).hexdigest()
        d49_text = '{"release": "d49"}'
        d59_text = '{"release": "d59", "validated": true}'
        d49_sha = hashlib.sha256(d49_text.encode()).hexdigest()
        d59_sha = hashlib.sha256(d59_text.encode()).hexdigest()
        options = launcher.LaunchOptions(runtime_profile="daily")
        with tempfile.TemporaryDirectory() as directory:
            release = self._release(directory, d49_text, sdk)
            with (
                unittest.mock.patch.object(
                    launcher, "selected_release", return_value=release
                ),
                unittest.mock.patch.object(launcher, "D48_SDK_SHA256", sdk_sha),
                unittest.mock.patch.object(
                    launcher,
                    "D48_RUNTIME_SHA256",
                    {"daily": d49_sha, "rccl-single": d49_sha},
                ),
                unittest.mock.patch.object(
                    launcher,
                    "RELEASE_D59_RUNTIME_SHA256",
                    {"daily": d59_sha, "rccl-single": d59_sha},
                ),
            ):
                self.assertTrue(launcher.qualified_negative_visibility(options))
                (release / "runtime.json").write_text(d59_text)
                self.assertTrue(launcher.qualified_negative_visibility(options))
                (release / "runtime.json").write_text('{"release": "unknown"}')
                self.assertFalse(launcher.qualified_negative_visibility(options))
                (release / "runtime.json").write_text(d59_text)
                (release / "rocm/lib/librocprofiler-sdk.so.1.4.2").write_bytes(b"other")
                self.assertFalse(launcher.qualified_negative_visibility(options))

    def test_broken_alias_refused(self):
        sdk = b"qualified-sdk"
        sdk_sha = hashlib.sha256(sdk).hexdigest()
        text = '{"release": "d59"}'
        digest = hashlib.sha256(text.encode()).hexdigest()
        options = launcher.LaunchOptions(runtime_profile="daily")
        with tempfile.TemporaryDirectory() as directory:
            release = self._release(directory, text, sdk, aliases=False)
            with (
                unittest.mock.patch.object(
                    launcher, "selected_release", return_value=release
                ),
                unittest.mock.patch.object(launcher, "D48_SDK_SHA256", sdk_sha),
                unittest.mock.patch.object(
                    launcher, "D48_RUNTIME_SHA256", {"daily": "x", "rccl-single": "x"}
                ),
                unittest.mock.patch.object(
                    launcher,
                    "RELEASE_D59_RUNTIME_SHA256",
                    {"daily": digest, "rccl-single": digest},
                ),
            ):
                self.assertFalse(launcher.qualified_negative_visibility(options))


if __name__ == "__main__":
    unittest.main()
