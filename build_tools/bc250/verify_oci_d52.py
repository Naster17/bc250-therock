#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Stream-check a saved OCI image against its config identity and layer diff IDs."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

IMAGE = "28ad36e62398875f0ad77112b1d0f2b1cd0b3b55560d9cc38c6908cada8caa06"


def digest(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        value.update(block)
    return value.hexdigest()


def verify(path, expected=IMAGE):
    with tarfile.open(path, "r:") as archive:
        members = {m.name.removeprefix("./"): m for m in archive if m.isfile()}

        def read(name):
            return archive.extractfile(members[name]).read()

        assert json.loads(read("oci-layout"))["imageLayoutVersion"] == "1.0.0"
        index = json.loads(read("index.json"))
        assert len(index["manifests"]) == 1
        seen = set()

        def blob(desc):
            algorithm, value = desc["digest"].split(":", 1)
            assert algorithm == "sha256"
            name = "blobs/sha256/" + value
            member = members[name]
            assert member.size == desc["size"]
            stream = archive.extractfile(member)
            assert digest(stream) == value, name
            seen.add(name)
            return member

        manifest_member = blob(index["manifests"][0])
        manifest = json.loads(archive.extractfile(manifest_member).read())
        assert manifest["schemaVersion"] == 2
        config_member = blob(manifest["config"])
        assert manifest["config"]["digest"] == "sha256:" + expected
        config = json.loads(archive.extractfile(config_member).read())
        assert config["architecture"] == "amd64" and config["os"] == "linux"
        diffs = config["rootfs"]["diff_ids"]
        assert len(diffs) == len(manifest["layers"])
        layers = []
        for desc, diff in zip(manifest["layers"], diffs):
            member = blob(desc)
            stream = archive.extractfile(member)
            media = desc["mediaType"]
            if media.endswith("+gzip") or media.endswith(".gzip"):
                stream = gzip.GzipFile(fileobj=stream)
            else:
                assert media in (
                    "application/vnd.oci.image.layer.v1.tar",
                    "application/vnd.docker.image.rootfs.diff.tar",
                ), media
            assert "sha256:" + digest(stream) == diff
            layers.append(
                dict(
                    digest=desc["digest"],
                    diff_id=diff,
                    bytes=desc["size"],
                    media_type=media,
                )
            )
        assert {name for name in members if name.startswith("blobs/")} == seen
    with path.open("rb") as stream:
        checksum = digest(stream)
    return dict(
        status="passed",
        image_id=expected,
        archive_sha256=checksum,
        archive_bytes=path.stat().st_size,
        layers=layers,
        manifest_digest=index["manifests"][0]["digest"],
        scope="OCI content/config/layer compressed and uncompressed identities; no load, execution or publication/license claim",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.archive)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in ["status", "image_id", "archive_bytes", "archive_sha256"]
            }
        )
    )


if __name__ == "__main__":
    main()
