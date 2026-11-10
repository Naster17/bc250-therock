#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Cross-check D52 raw metadata against separately verified image/source artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def verify(archive, image_report, source_report):
    with tarfile.open(archive) as content:
        data = {m.name: content.extractfile(m).read() for m in content if m.isfile()}
    get = lambda name: json.loads(data["evidence/" + name])
    image = get("source-d52-image-r1/result.json")
    assembly = get("source-d52-assembly-r2/result.json")
    failed = get("source-d52-assembly-r1/result.json")
    package = get("source-d52-package-r1/result.json")
    inputs = get("source-d52-inputs-r2/result.json")
    syntax = get("source-d52-package-r1/recipe-syntax.json")
    assert (
        image["status"]
        == assembly["status"]
        == package["status"]
        == inputs["status"]
        == syntax["status"]
        == "passed"
    )
    assert failed["status"] == "failed"
    assert image["verification"] == image_report and package["package"] == source_report
    assert (
        len(image_report["layers"]) == 14
        and image_report["image_id"]
        == "28ad36e62398875f0ad77112b1d0f2b1cd0b3b55560d9cc38c6908cada8caa06"
    )
    proof = assembly["assembly"]
    assert proof["parent_tree"] == "33a862732a52c8825a3d1f9c0b609d06fba0df65"
    assert (
        proof["parent_replay"]["trees"]["native"]
        == "3e3edf7f1de1fd1fe4730f9ec4ed30dd038b2112"
    )
    assert (
        proof["parent_blob_files"] == 21501
        and proof["materialized_dependency_locations"] == 69
    )
    assert proof["dependency_blob_files_with_duplicate_locations"] == 78598
    assert len(proof["checks"]) == 69 and len(proof["excluded_empty_paths"]) == 8
    assert sum(row["blob_files"] for row in proof["checks"]) == 78598
    gloo = next(c for c in proof["checks"] if c["path"] == "third_party/gloo")
    assert gloo["tree"] == "913f01fbb91a02ba2e30284fbe744e6b746e66f9"
    assert (
        source_report["source_files"] == 100099
        and not source_report["private_git_metadata_bundled"]
    )
    for phase in [image, assembly, package]:
        assert phase["samples"] and all(
            s["available_bytes"] >= 3 * 1024**3 and s["disk_free_bytes"] >= 5 * 1024**3
            for s in phase["samples"]
        )
    for profile, count in [("daily", 3239), ("rccl-single", 3241)]:
        p = inputs["profiles"][profile]
        assert (
            p["units"] == p["present_units"] == count and p["under_build_units"] == 260
        )
        assert not p["unresolved_units"] and len(p["declared_native_link_rules"]) == 38
        assert len(p["files"]) == count and all(
            row["present"] and len(row["sha256"]) == 64 for row in p["files"]
        )
    assert "--read-only" in syntax["command"] and "--network=none" in syntax["command"]
    return dict(
        status="passed",
        metadata_sha256=hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
        image=image_report,
        source=source_report,
        materialized_dependency_locations=69,
        compile_units=dict(daily=3239, rccl_single=3241),
        under_build_units_per_profile=260,
        declared_native_link_rules_per_profile=38,
        phases_seconds={
            name: row["seconds"]
            for name, row in [
                ("image_export", image),
                ("assembly", assembly),
                ("source_package", package),
            ]
        },
        min_host_available_bytes=min(
            s["available_bytes"]
            for row in [image, assembly, package]
            for s in row["samples"]
        ),
        min_disk_free_bytes=min(
            s["disk_free_bytes"]
            for row in [image, assembly, package]
            for s in row["samples"]
        ),
        scope="Separate real OCI/source archive checks plus raw assembly/resource/input metadata; no native rebuild/GPU/complete SDK/license proof",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--image-report", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(
        args.archive,
        json.loads(args.image_report.read_text()),
        json.loads(args.source_report.read_text()),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "status",
                    "metadata_sha256",
                    "materialized_dependency_locations",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
