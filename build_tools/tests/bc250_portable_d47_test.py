# SPDX-License-Identifier: MIT
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "bc250/portable_torch_d47.py"
spec = importlib.util.spec_from_file_location("portable_d47", SCRIPT)
portable = importlib.util.module_from_spec(spec)
spec.loader.exec_module(portable)


def package(root, member="LICENSE"):
    root.mkdir()
    with tarfile.open(root / "baseline.tar.gz", "w:gz") as output:
        entry = tarfile.TarInfo(member)
        entry.size = 1
        output.addfile(entry, io.BytesIO(b"x"))
    for name in ("0001-qualified-native-torch.patch", "0002-agent-exit-lifetime.patch"):
        (root / name).write_text("fixture")
    manifest = dict(
        format=1,
        baseline=portable.BASE,
        native=portable.NATIVE,
        source_head=portable.HEAD,
        payloads={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()
        },
    )
    manifest.update({name + "_tree": value for name, value in portable.TREES.items()})
    (root / "manifest.json").write_text(json.dumps(manifest))


class PortableTest(unittest.TestCase):
    def test_corrupt_payload_refused_before_destination_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package(root / "package")
            (root / "package/baseline.tar.gz").write_bytes(b"corrupt")
            with self.assertRaises(AssertionError):
                portable.replay(root / "package", root / "output")
            self.assertFalse((root / "output").exists())

    def test_wrong_attested_tree_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package(root / "package")
            path = root / "package/manifest.json"
            record = json.loads(path.read_text())
            record["native_tree"] = "0" * 40
            path.write_text(json.dumps(record))
            with self.assertRaises(AssertionError):
                portable.replay(root / "package", root / "output")

    def test_archive_escape_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package(root / "package", "../escape")
            with self.assertRaises(tarfile.FilterError):
                portable.replay(root / "package", root / "output")
            self.assertFalse((root / "escape").exists())

    def test_ambient_git_index_and_object_paths_not_inherited(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(
                os.environ,
                {
                    "GIT_DIR": "/not-a-repository",
                    "GIT_INDEX_FILE": str(root / "forbidden-index"),
                },
            ):
                portable.git(root, "init", "--quiet", "--object-format=sha1")
                self.assertEqual(
                    portable.git(root, "rev-parse", "--show-toplevel"), str(root)
                )
                (root / "owned").write_text("source")
                portable.git(root, "add", "owned")
                tree = portable.git(root, "write-tree")
                self.assertEqual(portable.entries(root, tree)[0]["path"], "owned")
            self.assertFalse((root / "forbidden-index").exists())


if __name__ == "__main__":
    unittest.main()
