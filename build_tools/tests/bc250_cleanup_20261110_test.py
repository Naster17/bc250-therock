# SPDX-License-Identifier: MIT
"""Deletion allowlist refuses source files, shared inodes and unrelated paths."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bc250"))
import cleanup_storage_20261110 as cleanup


class CleanupAllowlistTest(unittest.TestCase):
    def test_only_owned_unique_elf_object(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            obj = root / "owned.o"
            obj.write_bytes(b"\x7fELFfixture")
            with patch.object(cleanup, "OBJECT_ROOTS", [root]):
                self.assertEqual(cleanup.object_record(obj)["bytes"], 11)
                source = root / "source.cpp"
                source.write_bytes(obj.read_bytes())
                with self.assertRaises(AssertionError):
                    cleanup.object_record(source)
                with patch.object(cleanup, "OBJECT_ROOTS", [root / "different"]):
                    with self.assertRaises(AssertionError):
                        cleanup.object_record(obj)

    def test_hardlinked_or_symlinked_object_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            obj = root / "owned.o"
            obj.write_bytes(b"\x7fELFfixture")
            linked = root / "shared.o"
            linked.hardlink_to(obj)
            with patch.object(cleanup, "OBJECT_ROOTS", [root]):
                with self.assertRaises(AssertionError):
                    cleanup.object_record(obj)
                linked.unlink()
                linked.symlink_to(obj)
                with self.assertRaises(AssertionError):
                    cleanup.object_record(linked)

    def test_unknown_bytes_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            obj = root / "unknown.o"
            obj.write_bytes(b"not an ELF object")
            with patch.object(cleanup, "OBJECT_ROOTS", [root]):
                with self.assertRaises(AssertionError):
                    cleanup.object_record(obj)


if __name__ == "__main__":
    unittest.main()
