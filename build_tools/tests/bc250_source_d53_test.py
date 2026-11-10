# SPDX-License-Identifier: MIT
"""Payload identity, duplicate-member and link-boundary regressions."""
import gzip
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bc250"))
from external_kit_d53 import create, verify, inventory
from admit_clean_build_d53 import admission


class ExternalKitTest(unittest.TestCase):
    def test_build_admission_keeps_reserve(self):
        budget = dict(source=3, cache=20, reserve=5)
        self.assertFalse(admission(27, budget)["admitted"])
        self.assertTrue(admission(28, budget)["admitted"])
        self.assertEqual(admission(12, budget)["shortfall_bytes"], 16)

    def fixture(self, root):
        source = root / "source"
        source.mkdir()
        (source / "file").write_bytes(b"original\r\n")
        os.link(source / "file", source / "hardlink")
        (source / "relative").symlink_to("file")
        (source / "external").symlink_to("/outside/input")
        archive = root / "kit.tar.gz"
        with archive.open("wb") as stream:
            create(dict(rocm=str(source)), stream)
        return archive

    def rewrite(self, original, output, change):
        with tarfile.open(original) as source, tarfile.open(output, "w:gz") as dest:
            for member in source:
                data = source.extractfile(member).read() if member.isfile() else None
                member, data = change(member, data)
                dest.addfile(member, io.BytesIO(data) if data is not None else None)

    def test_real_payload_and_hardlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = self.fixture(Path(directory))
            report = verify(archive)
            self.assertEqual(report["file_bytes"], 20)
            self.assertEqual(
                report["external_or_unresolved_links"],
                [dict(path="rocm/external", target="/outside/input")],
            )

    def test_deduplicated_payload_and_fifo(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "a").write_bytes(b"same bytes")
            (source / "b").write_bytes(b"same bytes")
            (source / "c").write_bytes(b"other data")
            os.mkfifo(source / "closed-pipe")
            archive = root / "deduplicated.tar.gz"
            with archive.open("wb") as stream:
                create({"source": source}, stream, deduplicate=True)
            report = verify(archive)
            self.assertEqual(report["file_bytes"], 30)
            with tarfile.open(archive) as src:
                self.assertEqual(src.getmember("source/b").linkname, "source/a")
                self.assertTrue(src.getmember("source/closed-pipe").isfifo())

            def change(member, data):
                if member.name == "source/b":
                    member.linkname = "source/c"
                return member, data

            changed = root / "wrong-reference.tar.gz"
            self.rewrite(archive, changed, change)
            with self.assertRaises(AssertionError):
                verify(changed)

    def test_manifest_admission_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = self.fixture(Path(directory))
            with self.assertRaises(AssertionError):
                verify(archive, max_manifest_bytes=16)
            self.assertEqual(
                verify(archive, max_manifest_bytes=4096)["status"], "passed"
            )

    def test_complete_gzip_published_before_source_recheck(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "file").write_bytes(b"content")
            sink = io.BytesIO()
            output = io.BufferedWriter(sink, buffer_size=1024**2)
            calls = 0

            def checked_inventory(roots):
                nonlocal calls
                calls += 1
                if calls == 2:
                    self.assertTrue(
                        gzip.decompress(sink.getvalue()).endswith(b"\0" * 1024)
                    )
                return inventory(roots)

            with patch("external_kit_d53.inventory", side_effect=checked_inventory):
                create(dict(source=str(root)), output)
            output.detach()

    def test_content_change_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self.fixture(root)

            def change(member, data):
                if member.name == "rocm/file":
                    data = b"corrupt!\r\n"
                return member, data

            self.rewrite(archive, root / "bad.tar.gz", change)
            with self.assertRaises(AssertionError):
                verify(root / "bad.tar.gz")

    def test_duplicate_member_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self.fixture(root)
            changed = root / "duplicate.tar.gz"
            with tarfile.open(archive) as src, tarfile.open(changed, "w:gz") as dst:
                for member in src:
                    data = src.extractfile(member).read() if member.isfile() else None
                    dst.addfile(member, io.BytesIO(data) if data is not None else None)
                    if member.name == "rocm/file":
                        dst.addfile(member, io.BytesIO(data))
            with self.assertRaises(AssertionError):
                verify(changed)

    def test_gzip_trailer_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self.fixture(root)
            data = bytearray(archive.read_bytes())
            data[-8] ^= 1
            archive.write_bytes(data)
            with self.assertRaises(OSError):
                verify(archive)

    def test_permission_change_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self.fixture(root)

            def change(member, data):
                if member.name == "rocm/file":
                    member.mode ^= 0o111
                return member, data

            self.rewrite(archive, root / "mode.tar.gz", change)
            with self.assertRaises(AssertionError):
                verify(root / "mode.tar.gz")

    def test_missing_member_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self.fixture(root)

            def change(member, data):
                if member.name == "manifest.json":
                    manifest = json.loads(data)
                    manifest["entries"].append(
                        dict(path="rocm/missing", kind="directory", mode=0o755)
                    )
                    data = json.dumps(manifest).encode()
                    member.size = len(data)
                return member, data

            self.rewrite(archive, root / "missing.tar.gz", change)
            with self.assertRaises(AssertionError):
                verify(root / "missing.tar.gz")


if __name__ == "__main__":
    unittest.main()
