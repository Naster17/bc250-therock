#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check raw D50 host evidence; full source replay remains a separate command."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def verify(archive, package_receipt):
    with tarfile.open(archive) as content:
        data = {m.name: content.extractfile(m).read() for m in content if m.isfile()}
    get = lambda name: json.loads(data[name])
    sha = lambda value: hashlib.sha256(value).hexdigest()
    record = get("evidence/result.json")
    replay = get("evidence/replay.json")
    assert (
        record["status"] == replay["status"] == "passed" and record["replay"] == replay
    )
    assert (
        record["driver_sha256"]
        == sha(data["package/portable_dependencies_d50.py"])
        == package_receipt["driver_sha256"]
    )
    assert (
        record["manifest_sha256"]
        == replay["manifest_sha256"]
        == sha(data["package/manifest.json"])
        == package_receipt["manifest_sha256"]
    )
    assert replay["parent_manifest_sha256"] == sha(data["package/parent-manifest.json"])
    assert (
        replay["nodes"],
        replay["available_nodes"],
        replay["unique_payloads"],
        replay["blob_files"],
        replay["license_files"],
    ) == (77, 69, 64, 75522, 107)
    assert (
        len(replay["missing"]) == 8 and not replay["complete_recursive_source_closure"]
    )
    for sample in record["samples"]:
        assert (
            sample["available_bytes"] >= 3 * 1024**3
            and sample["disk_free_bytes"] >= 5 * 1024**3
        )
    gloo_sha = sha(data["evidence/metadata/retained-gloo-cuda.cu"])
    assert gloo_sha == replay["gloo_cuda_sha256"]
    for profile, name in [
        ("daily", "pytorch-quant-b052"),
        ("rccl-single", "pytorch-quant-b053"),
    ]:
        retained = record["retained_builds"][profile]
        cmake = data["evidence/metadata/" + name + "-CMakeCache.txt"]
        version = data["evidence/metadata/" + name + "-version.py"]
        assert retained["cmake_sha256"] == sha(cmake)
        assert (
            retained["version_sha256"] == sha(version)
            and retained["version_text"] == version.decode()
        )
        assert retained["gloo_cuda_sha256"] == gloo_sha
        values = {
            line.split(":", 1)[0]: line.partition("=")[2]
            for line in cmake.decode().splitlines()
            if ":" in line and "=" in line and not line.startswith(("#", "//"))
        }
        assert all(values[key] == value for key, value in retained["flags"].items())
        assert (
            values["USE_FLASH_ATTENTION"]
            == values["USE_MEM_EFF_ATTENTION"]
            == values["USE_MSLK"]
            == "0"
        )
        assert values["USE_GLOO"] == values["USE_TENSORPIPE"] == "1"
    assert (
        record["sdk_sha256"]
        == "2a975370ba4eb1c2fe39e406ff36b175f554a49d0aa19febbd326f641fcb590b"
    )
    return dict(
        status="passed",
        scope="metadata/source/retained-input consistency; complete dependency-tree replay is separate",
        archive_sha256=sha(Path(archive).read_bytes()),
        seconds=record["seconds"],
        min_host_available_bytes=min(x["available_bytes"] for x in record["samples"]),
        min_disk_free_bytes=min(x["disk_free_bytes"] for x in record["samples"]),
        **{
            key: replay[key]
            for key in [
                "nodes",
                "available_nodes",
                "unique_payloads",
                "blob_files",
                "license_files",
                "complete_recursive_source_closure",
                "gloo_overlay_tree",
                "gloo_cuda_sha256",
            ]
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--package-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.archive, json.loads(args.package_receipt.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
