# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import io
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export_source_archive import TarArchiveWriter, create_archive


class ExportSourceArchiveTest(unittest.TestCase):
    def test_other_repository_with_special_filenames(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            filenames = ["normal.txt", "日本語.txt", "line\nbreak.txt"]
            for filename in filenames:
                (repo / filename).write_text(filename)
            git = ["git", "-C", str(repo)]
            subprocess.run([*git, "add", "."], check=True)
            subprocess.run(
                [
                    *git,
                    "-c",
                    "user.name=Archive Test",
                    "-c",
                    "user.email=archive@example.invalid",
                    "commit",
                    "-qm",
                    "test sources",
                ],
                check=True,
            )
            origin = "https://example.invalid/source.git"
            subprocess.run([*git, "remote", "add", "origin", origin], check=True)
            revision = subprocess.check_output([*git, "rev-parse", "HEAD"]).strip()
            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode="w") as archive:
                create_archive(TarArchiveWriter(archive), repo, None)
            stream.seek(0)
            with tarfile.open(fileobj=stream) as archive:
                self.assertEqual(
                    set(archive.getnames()), {*filenames, "GIT_REVISION", "GIT_ORIGIN"}
                )
                self.assertEqual(archive.extractfile("GIT_REVISION").read(), revision)
                self.assertEqual(
                    archive.extractfile("GIT_ORIGIN").read(), origin.encode()
                )
                for filename in filenames:
                    self.assertEqual(
                        archive.extractfile(filename).read(), filename.encode()
                    )


if __name__ == "__main__":
    unittest.main()
