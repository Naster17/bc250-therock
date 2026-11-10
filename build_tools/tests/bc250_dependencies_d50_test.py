# SPDX-License-Identifier: MIT
"""Portable dependency replay binds commit, tree and filesystem boundaries."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tarfile
import io
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / "bc250"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location(
    "dependencies_d50", TOOLS / "portable_dependencies_d50.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class DependencyReplayTest(unittest.TestCase):
    def fixture(self, root, with_symlink=False):
        repo = root / "original"
        repo.mkdir()
        m.git(repo, "init", "--quiet")
        (repo / "LICENSE").write_text("Owned license fixture\n")
        (repo / "code.txt").write_text("original\n")
        if with_symlink:
            (repo / "dir/sub").mkdir(parents=True)
            (repo / "dir/sub/link").symlink_to("../../")
        m.git(repo, "add", "--all")
        m.git(
            repo,
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        )
        pin = m.git(repo, "rev-parse", "HEAD")
        package = root / "package"
        (package / "objects").mkdir(parents=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "archive",
                "--format=tar.gz",
                "--output=" + str(package / "objects" / (pin + ".tar.gz")),
                pin,
            ],
            check=True,
            env=m.git_environment(),
        )
        data = subprocess.check_output(
            ["git", "-C", str(repo), "cat-file", "commit", pin], env=m.git_environment()
        )
        (package / "objects" / (pin + ".commit")).write_bytes(data)
        entries = m.entries(repo, pin)
        record = dict(
            tree=m.git(repo, "rev-parse", pin + "^{tree}"),
            archive_sha256=m.sha(package / "objects" / (pin + ".tar.gz")),
            commit_sha256=m.sha(package / "objects" / (pin + ".commit")),
            gitlinks=[],
            blob_files=sum(e["kind"] == "blob" for e in entries),
            license_files=[e for e in entries if e["path"] == "LICENSE"],
        )
        return pin, package, record

    def test_tree_and_commit_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pin, package, record = self.fixture(root)
            m.verify_tree(package, pin, record, root / "valid")
            data = package / "objects" / (pin + ".commit")
            data.write_bytes(data.read_bytes() + b"changed")
            with self.assertRaises(AssertionError):
                m.verify_tree(package, pin, record, root / "invalid")

    def test_wrong_tree_cannot_be_relabelled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pin, package, record = self.fixture(root)
            record["tree"] = "0" * 40
            with self.assertRaises(AssertionError):
                m.verify_tree(package, pin, record, root / "invalid")

    def test_ambient_git_environment_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pin, package, record = self.fixture(root)
            with patch.dict(
                os.environ,
                {
                    "GIT_DIR": str(root / "foreign"),
                    "GIT_INDEX_FILE": str(root / "foreign-index"),
                    "GIT_OBJECT_DIRECTORY": str(root / "foreign-objects"),
                },
            ):
                m.verify_tree(package, pin, record, root / "valid")
            self.assertFalse((root / "foreign-index").exists())
            self.assertFalse((root / "foreign-objects").exists())

    def test_symlink_target_bytes_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pin, package, record = self.fixture(root, with_symlink=True)
            m.verify_tree(package, pin, record, root / "valid")
            self.assertEqual(os.readlink(root / "valid/dir/sub/link"), "../../")

    def test_payload_change_with_updated_sha_still_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pin, package, record = self.fixture(root)
            archive = package / "objects" / (pin + ".tar.gz")
            with tarfile.open(archive) as content:
                records = [
                    (item, content.extractfile(item).read() if item.isfile() else None)
                    for item in content
                ]
            with tarfile.open(archive, "w:gz") as content:
                for item, data in records:
                    if item.name == "code.txt":
                        data = b"altered source\n"
                        item.size = len(data)
                    content.addfile(
                        item, io.BytesIO(data) if data is not None else None
                    )
            record["archive_sha256"] = m.sha(archive)
            with self.assertRaises(AssertionError):
                m.verify_tree(package, pin, record, root / "invalid")

    def test_private_git_and_parent_paths_refused(self):
        for name in ["../escape", "/absolute", "ok/.git/config", ""]:
            with self.subTest(name=name), self.assertRaises(AssertionError):
                m.safe_path(name)


if __name__ == "__main__":
    unittest.main()
