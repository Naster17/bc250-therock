# SPDX-License-Identifier: MIT
"""Owned fixed-world checkpoint header/path/hash admission, independent of Torch."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bc250"))
from distributed_d10_checkpoint import canonical, digest, local_part


class CheckpointAdmissionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "rank-0.pt").write_bytes(b"owned checkpoint")
        self.config = dict(case="owned")
        self.cohort = "1" * 64
        self.commit = dict(
            schema=1,
            config=self.config,
            cohort=self.cohort,
            world=2,
            next_attempt=3,
            parts=[
                dict(
                    rank=i,
                    file=f"rank-{i}.pt",
                    bytes=16,
                    sha256=digest(self.root / "rank-0.pt"),
                )
                for i in [0, 1]
            ],
        )

    def tearDown(self):
        self.tmp.cleanup()

    def check(self, commit):
        return local_part(commit, self.root, self.config, self.cohort, 0, 2, 3)

    def test_valid_own_part_without_shared_filesystem(self):
        self.assertEqual(self.check(self.commit), self.root / "rank-0.pt")
        self.assertFalse((self.root / "rank-1.pt").exists())

    def test_header_refusals(self):
        for key, value in [
            ("schema", 2),
            ("config", {}),
            ("cohort", "2" * 64),
            ("world", 3),
            ("next_attempt", 4),
            ("parts", []),
        ]:
            bad = copy.deepcopy(self.commit)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check(bad)
        with self.assertRaises(ValueError):
            self.check(None)

    def test_paths_rank_and_hash_fields(self):
        for field, value in [
            ("rank", 1),
            ("file", "../rank-0.pt"),
            ("bytes", 0),
            ("bytes", "16"),
            ("sha256", "bad"),
        ]:
            bad = copy.deepcopy(self.commit)
            bad["parts"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.check(bad)

    def test_truncation_and_symlink(self):
        original = (self.root / "rank-0.pt").read_bytes()
        (self.root / "rank-0.pt").write_bytes(b"short")
        with self.assertRaises(ValueError):
            self.check(self.commit)
        (self.root / "payload").write_bytes(original)
        (self.root / "rank-0.pt").unlink()
        (self.root / "rank-0.pt").symlink_to("payload")
        with self.assertRaises(ValueError):
            self.check(self.commit)

    def test_canonical_identity(self):
        self.assertEqual(canonical(dict(a=1, b=2)), canonical(dict(b=2, a=1)))
        self.assertNotEqual(canonical(dict(a=1)), canonical(dict(a=2)))


if __name__ == "__main__":
    unittest.main()
