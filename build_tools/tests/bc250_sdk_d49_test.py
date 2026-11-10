# SPDX-License-Identifier: MIT
"""Fail-closed native SDK identity and lock-time validation regressions."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "bc250/bc250_torch.py"
spec = importlib.util.spec_from_file_location("sdk_d49_launcher", SCRIPT)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class SDKIdentityTest(unittest.TestCase):
    def fixture(self, root):
        (root / "rocm/lib").mkdir(parents=True)
        (root / "runtime.json").write_text("baseline")
        lib = root / "rocm/lib/librocprofiler-sdk.so.1.4.2"
        lib.write_text("qualified fixture")
        for alias in ("librocprofiler-sdk.so", "librocprofiler-sdk.so.1"):
            (lib.parent / alias).symlink_to(lib.name)
        sha = launcher.hashlib.sha256
        return (
            sha(lib.read_bytes()).hexdigest(),
            sha((root / "runtime.json").read_bytes()).hexdigest(),
        )

    def test_identity_changes_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sdk, runtime = self.fixture(root)
            options = launcher.LaunchOptions(environment={"HIP_VISIBLE_DEVICES": "-1"})
            with patch.object(
                launcher, "selected_release", return_value=root
            ), patch.object(launcher, "D48_SDK_SHA256", sdk), patch.dict(
                launcher.D48_RUNTIME_SHA256, daily=runtime
            ):
                launcher.validate_options(options)
                lib = root / "rocm/lib/librocprofiler-sdk.so.1.4.2"
                lib.write_text("old or corrupted SDK")
                with self.assertRaisesRegex(ValueError, "recorded ROCr import crash"):
                    launcher.validate_options(options)
                lib.write_text("qualified fixture")
                (root / "runtime.json").write_text("other cohort")
                self.assertFalse(launcher.qualified_negative_visibility(options))

    def test_alias_and_missing_payload_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sdk, runtime = self.fixture(root)
            with patch.object(
                launcher, "selected_release", return_value=root
            ), patch.object(launcher, "D48_SDK_SHA256", sdk), patch.dict(
                launcher.D48_RUNTIME_SHA256, daily=runtime
            ):
                alias = root / "rocm/lib/librocprofiler-sdk.so.1"
                alias.unlink()
                alias.write_text("different provider")
                self.assertFalse(
                    launcher.qualified_negative_visibility(launcher.LaunchOptions())
                )
                alias.unlink()
                self.assertFalse(
                    launcher.qualified_negative_visibility(launcher.LaunchOptions())
                )

    def test_build_revalidates_after_parse(self):
        options = launcher.LaunchOptions(environment={"HIP_VISIBLE_DEVICES": "-1"})
        with patch.object(
            launcher, "qualified_negative_visibility", return_value=False
        ):
            with self.assertRaisesRegex(ValueError, "recorded ROCr import crash"):
                launcher.build_command(
                    Path("/unused"),
                    {},
                    Path("/unused"),
                    Path("/unused"),
                    options,
                    [],
                    Path("/unused"),
                )


if __name__ == "__main__":
    unittest.main()
