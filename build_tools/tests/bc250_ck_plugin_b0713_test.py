# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Local tests for the B07.13 activation wrapper integrity logic.

Mocks the host-specific launcher/migraphx dependencies so check_package and the
identity constants can be exercised on the workstation without /opt or root.
"""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "bc250/bc250_ck_b0713.py"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_wrapper():
    # Mock the host-specific modules the wrapper imports at module level.
    migraphx = types.ModuleType("bc250_migraphx")
    migraphx.sha = _sha
    migraphx.trusted = lambda path: None  # root-ownership check is host-only
    torch_mod = types.ModuleType("bc250_torch")
    for name in (
        "exit_status",
        "parse_launcher_arguments",
        "prepare_python_arguments",
        "run_process",
        "runtime_lock",
        "read_context",
        "selected_release",
    ):
        setattr(torch_mod, name, lambda *a, **k: None)
    sys.modules["bc250_migraphx"] = migraphx
    sys.modules["bc250_torch"] = torch_mod
    spec = importlib.util.spec_from_file_location("bc250_ck_b0713", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IdentityConstantsTest(unittest.TestCase):
    def test_pins_match_b0712_artifact(self):
        wrapper = _load_wrapper()
        self.assertEqual(wrapper.NAME, "bc250-miopen-dl-gfx1013-b0712")
        self.assertEqual(
            wrapper.ARCHIVE_SHA,
            "a0d47ed4ba4df354cc7ea898a285eefa051f49693667386567ea392f66ddd91e",
        )
        self.assertEqual(
            wrapper.MANIFEST_SHA,
            "52db5d4262bd29e6307b13a5862cc0652f0e0b6f9c8509761d03d39965e02846",
        )
        self.assertEqual(wrapper.BASE, Path("/opt/bc250-stack/sdks"))
        self.assertEqual(wrapper.REGISTRY, wrapper.BASE / "miopen-dl.json")


class CheckPackageTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.wrapper = _load_wrapper()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "pkg"
        self.root.mkdir()
        (self.root / "lib").mkdir()
        (self.root / "lib" / "plugin.so").write_bytes(b"plugin-bytes")
        (self.root / "README.md").write_bytes(b"readme")
        files = {
            "lib/plugin.so": _sha(self.root / "lib/plugin.so"),
            "README.md": _sha(self.root / "README.md"),
        }
        self.manifest = {"format": 2, "files": files}
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))
        # Pin the wrapper to this synthetic manifest for the integrity test.
        self.wrapper.MANIFEST_SHA = _sha(self.root / "manifest.json")
        self.record = {"manifest_sha256": self.wrapper.MANIFEST_SHA}

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_package_passes(self):
        manifest = self.wrapper.check_package(self.root, self.record)
        self.assertEqual(manifest["format"], 2)

    def test_payload_drift_refused(self):
        (self.root / "README.md").write_bytes(b"readme-drift")
        with self.assertRaisesRegex(ValueError, "payload identity mismatch"):
            self.wrapper.check_package(self.root, self.record)

    def test_extra_file_refused(self):
        (self.root / "unexpected.py").write_bytes(b"# extra")
        with self.assertRaisesRegex(ValueError, "file set mismatch"):
            self.wrapper.check_package(self.root, self.record)

    def test_manifest_mismatch_refused(self):
        self.wrapper.MANIFEST_SHA = "0" * 64
        with self.assertRaisesRegex(ValueError, "manifest identity mismatch"):
            self.wrapper.check_package(self.root, self.record)

    def test_record_manifest_mismatch_refused(self):
        self.record["manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "manifest identity mismatch"):
            self.wrapper.check_package(self.root, self.record)


if __name__ == "__main__":
    unittest.main()
