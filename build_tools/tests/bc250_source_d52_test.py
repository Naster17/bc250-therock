# SPDX-License-Identifier: MIT
"""Image identity/diff validation and exact materialized-source checks."""
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[1] / "bc250"
sys.path.insert(0, str(TOOLS))
from assemble_source_d52 import verify_materialized, fresh_recipe, blob_id
from inventory_generated_d52_on_host import native_link_rules
from source_kit_d52 import create as create_kit, extract as extract_kit
from verify_oci_d52 import verify


def sha(data):
    return hashlib.sha256(data).hexdigest()


class SourceAssemblyTest(unittest.TestCase):
    def test_exact_file_symlink_and_nested_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "file").write_bytes(b"content")
            (root / "link").symlink_to("file")
            (root / "nested").mkdir()
            (root / "nested/owned").write_text("separate dependency")
            expected = [
                dict(
                    path="file",
                    mode="100644",
                    kind="blob",
                    oid=blob_id(root / "file", "100644"),
                ),
                dict(
                    path="link",
                    mode="120000",
                    kind="blob",
                    oid=blob_id(root / "link", "120000"),
                ),
                dict(path="nested", mode="160000", kind="commit", oid="0" * 40),
            ]
            self.assertEqual(verify_materialized(root, expected), 2)
            (root / "extra").write_text("untracked content")
            with self.assertRaises(AssertionError):
                verify_materialized(root, expected)

    def test_mode_and_bytes_drift_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "file").write_bytes(b"original")
            expected = [
                dict(
                    path="file",
                    mode="100644",
                    kind="blob",
                    oid=blob_id(root / "file", "100644"),
                )
            ]
            (root / "file").chmod(0o755)
            with self.assertRaises(AssertionError):
                verify_materialized(root, expected)
            (root / "file").chmod(0o644)
            (root / "file").write_bytes(b"changed")
            with self.assertRaises(AssertionError):
                verify_materialized(root, expected)

    def test_working_tree_eol_bytes_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "copy"
            proof = Path(directory) / "proof"
            root.mkdir()
            proof.mkdir()
            (root / "template").write_bytes(b"a\r\nb\r\n")
            (proof / "template").write_bytes(b"a\r\nb\r\n")
            canonical = b"a\nb\n"
            expected = [
                dict(
                    path="template",
                    mode="100644",
                    kind="blob",
                    oid=hashlib.sha1(b"blob 4\0" + canonical).hexdigest(),
                )
            ]
            with self.assertRaises(AssertionError):
                verify_materialized(root, expected)
            self.assertEqual(verify_materialized(root, expected, reference=proof), 1)
            (root / "template").write_bytes(canonical)
            with self.assertRaises(AssertionError):
                verify_materialized(root, expected, reference=proof)

    def test_link_target_at_end_before_colon(self):
        ninja = (
            "build lib/libtorch.so: CXX_SHARED_LIBRARY_LINKER__torch obj.o\n"
            + "build lib/libuv.a: CXX_STATIC_LIBRARY_LINKER__uv uv.o\n"
            + "build libtorch.so: phony lib/libtorch.so\n"
        )
        self.assertEqual(len(native_link_rules(ninja)), 2)

    def test_gitless_archive_roundtrip_preserves_working_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assembly = root / "assembled"
            (assembly / "torch/deep/sub").mkdir(parents=True)
            (assembly / "torch/.git").mkdir()
            (assembly / "torch/.git/private").write_text("must stay private")
            (assembly / "torch/template").write_bytes(b"a\r\nb\r\n")
            (assembly / "torch/deep/sub/link").symlink_to("../../")
            (assembly / "recipes").mkdir()
            (assembly / "recipes/recipe.sh").write_text("#!/bin/sh\nexit 0\n")
            (assembly / "assembly.json").write_text(
                json.dumps(
                    dict(
                        status="passed",
                        parent_blob_files=2,
                        dependency_blob_files_with_duplicate_locations=0,
                    )
                )
            )
            result = create_kit(assembly, root / "source.tar.gz")
            self.assertEqual(result["source_files"], 2)
            extract_kit(root / "source.tar.gz", root / "unpacked")
            source = root / "unpacked/torch-source-d52-r1/torch"
            self.assertFalse((source / ".git").exists())
            self.assertEqual((source / "template").read_bytes(), b"a\r\nb\r\n")
            self.assertEqual(os.readlink(source / "deep/sub/link"), "../../")

    def test_fresh_recipes_do_not_reuse_old_cache(self):
        for suffix in ["b052", "b053"]:
            text = (TOOLS / ("build_torch_" + suffix + ".sh")).read_text()
            fresh = fresh_recipe(text, rccl=suffix == "b053")
            self.assertNotIn("cp build/CMakeCache.txt", fresh)
            self.assertIn(
                "test ! -e /workspace/pytorch-work/build/CMakeCache.txt", fresh
            )
            self.assertIn("python -m pip install -e . -v --no-build-isolation", fresh)
            if suffix == "b053":
                self.assertIn("/workspace/rccl/lib/cmake/rccl", fresh)
                self.assertNotIn("/home/nik/", fresh)


class OCIValidationTest(unittest.TestCase):
    def fixture(self, path, bad_diff=False, extra=False):
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode="w") as content:
            member = tarfile.TarInfo("owned")
            member.size = 5
            content.addfile(member, io.BytesIO(b"hello"))
        uncompressed = raw.getvalue()
        layer = gzip.compress(uncompressed, mtime=0)
        config = json.dumps(
            dict(
                architecture="amd64",
                os="linux",
                rootfs=dict(
                    type="layers",
                    diff_ids=[
                        "sha256:" + ("0" * 64 if bad_diff else sha(uncompressed))
                    ],
                ),
            )
        ).encode()
        descriptor = lambda data, media: dict(
            mediaType=media, digest="sha256:" + sha(data), size=len(data)
        )
        manifest = json.dumps(
            dict(
                schemaVersion=2,
                config=descriptor(config, "application/vnd.oci.image.config.v1+json"),
                layers=[
                    descriptor(layer, "application/vnd.oci.image.layer.v1.tar+gzip")
                ],
            )
        ).encode()
        blobs = {sha(config): config, sha(layer): layer, sha(manifest): manifest}
        if extra:
            blobs[sha(b"extra")] = b"extra"
        files = {
            "oci-layout": json.dumps(dict(imageLayoutVersion="1.0.0")).encode(),
            "index.json": json.dumps(
                dict(
                    schemaVersion=2,
                    manifests=[
                        descriptor(
                            manifest, "application/vnd.oci.image.manifest.v1+json"
                        )
                    ],
                )
            ).encode(),
            **{"blobs/sha256/" + key: value for key, value in blobs.items()},
        }
        with tarfile.open(path, "w") as archive:
            for name, data in files.items():
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        return sha(config)

    def test_valid_gzip_and_config_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.tar"
            expected = self.fixture(path)
            self.assertEqual(verify(path, expected)["image_id"], expected)
            with self.assertRaises(AssertionError):
                verify(path, "0" * 64)

    def test_inconsistent_uncompressed_diff_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.tar"
            expected = self.fixture(path, bad_diff=True)
            with self.assertRaises(AssertionError):
                verify(path, expected)

    def test_unreachable_blob_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.tar"
            expected = self.fixture(path, extra=True)
            with self.assertRaises(AssertionError):
                verify(path, expected)


if __name__ == "__main__":
    unittest.main()
