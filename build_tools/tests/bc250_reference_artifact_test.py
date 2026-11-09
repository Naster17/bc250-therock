# SPDX-License-Identifier: MIT
"""Raw staged references must refuse corruption, path escape and unsafe capacity."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

BASE = Path(__file__).parents[1] / "bc250"
spec = importlib.util.spec_from_file_location("artifact_d35", BASE / "artifact_d35.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReferenceArtifactTest(unittest.TestCase):
    def test_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = bytes(range(256)) * 8193
            item = module.write_blob(root, "test.raw", data)
            module.verify_blob(root, item)
            self.assertEqual(module.read_blob(root, item), data)
            with self.assertRaises(FileExistsError):
                module.write_blob(root, "test.raw", b"bad")
            self.assertEqual(module.read_blob(root, item), data)

    def test_same_size_corruption_and_truncation_refuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            item = module.write_blob(root, "test.raw", b"reference")
            (root / "test.raw").write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "digest"):
                module.verify_blob(root, item)
            with self.assertRaisesRegex(ValueError, "digest"):
                module.read_blob(root, item)
            (root / "test.raw").write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "size"):
                module.verify_blob(root, item)

    def test_escape_and_symlink_refuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            canary = root / "canary"
            canary.write_bytes(b"unchanged")
            (root / "link.raw").symlink_to(canary)
            for name in ["../canary.raw", "/tmp/canary.raw", "link.raw"]:
                with self.assertRaises(ValueError):
                    module.write_blob(root, name, b"damaged")
            self.assertEqual(canary.read_bytes(), b"unchanged")

    def test_disk_reserve_and_payload(self):
        parameters = 352957440
        for producing, width in [(True, 12), (False, 4)]:
            limit = 5 * module.GIB + parameters * width
            with self.assertRaises(ValueError):
                module.admit_storage(limit - 1, parameters, producing=producing)
            self.assertEqual(
                module.admit_storage(limit, parameters, producing=producing)[
                    "required_payload_bytes"
                ],
                parameters * width,
            )

    def test_manifest_integrity_shape_alias_and_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            initial = module.write_blob(root, "initial.raw", b"\0" * 8)
            grad = module.write_blob(root, "gradient.raw", b"\0" * 16)
            record = dict(
                schema=1,
                status="passed",
                config={"fixture": 1},
                parameters=2,
                cuda_initialized=False,
                CPU_FP64_loss=1.0,
                sources={},
                tensors=[dict(name="p", shape=[2], initial=initial, gradient=grad)],
            )
            path = root / "manifest.json"

            def publish():
                path.write_text(json.dumps(record))
                return module.digest(path)

            pin = publish()
            self.assertEqual(
                module.load_manifest(path, pin, {"fixture": 1}, 2, check_sources=False),
                record,
            )
            with self.assertRaises(ValueError):
                module.load_manifest(
                    path, "0" * 64, {"fixture": 1}, 2, check_sources=False
                )
            record["tensors"][0]["shape"] = [3]
            with self.assertRaises(ValueError):
                module.load_manifest(
                    path, publish(), {"fixture": 1}, 2, check_sources=False
                )
            record["tensors"][0]["shape"] = [2]
            record["tensors"].append(record["tensors"][0])
            with self.assertRaises(ValueError):
                module.load_manifest(
                    path, publish(), {"fixture": 1}, 2, check_sources=False
                )
