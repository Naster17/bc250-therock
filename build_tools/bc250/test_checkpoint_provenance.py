# SPDX-License-Identifier: MIT
"""CPU-only checkpoint metadata compatibility and failure-boundary contracts."""

from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest

import bc250_checkpoint_provenance as provenance


class CheckpointProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.record = dict(
            schema=1, fingerprint="f" * 64, identity={"preset_sha256": "a" * 64}
        )
        self.run = {
            provenance.MARKER: provenance.write_checkpoint(self.root, self.record)
        }

    def read(self):
        return provenance.read_checkpoint(
            self.root, self.run, self.record["fingerprint"]
        )

    def test_roundtrip_and_same_identity(self):
        self.assertEqual(self.read(), self.record)
        provenance.check_resume(self.read(), self.record)

    def test_legacy_requires_absent_binding_and_sidecar(self):
        self.run.clear()
        with self.assertRaisesRegex(ValueError, "Orphan"):
            self.read()
        (self.root / provenance.FILENAME).unlink()
        self.assertIsNone(self.read())
        stream = io.StringIO()
        with redirect_stderr(stream):
            provenance.check_resume(None, self.record)
        self.assertIn("Legacy checkpoint", stream.getvalue())
        self.assertIn("unverified", stream.getvalue())

    def test_missing_corrupt_and_future_schema_refused(self):
        sidecar = self.root / provenance.FILENAME
        sidecar.unlink()
        with self.assertRaisesRegex(ValueError, "Missing or malformed"):
            self.read()
        sidecar.write_text("{")
        with self.assertRaisesRegex(ValueError, "Missing or malformed"):
            self.read()
        for schema in (True, 2):
            record = {**self.record, "schema": schema}
            self.run[provenance.MARKER] = provenance.write_checkpoint(self.root, record)
            with self.assertRaisesRegex(ValueError, "corrupted"):
                self.read()
        self.run[provenance.MARKER] = provenance.write_checkpoint(
            self.root, self.record
        )
        sidecar.write_text(json.dumps({**self.record, "fingerprint": "0" * 64}))
        with self.assertRaisesRegex(ValueError, "corrupted"):
            self.read()

    def test_changed_identity_refused_without_mutation(self):
        before = (self.root / provenance.FILENAME).read_bytes()
        changed = {**self.record, "identity": {"preset_sha256": "b" * 64}}
        with self.assertRaisesRegex(ValueError, "numerical runtime mismatch"):
            provenance.check_resume(self.record, changed)
        self.assertEqual((self.root / provenance.FILENAME).read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
