#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify portable selected-build input evidence without Torch, GPU or network."""
import argparse
import json
from pathlib import Path
import tomllib

from build_inputs_d51 import sha, classify, build_settings

SOURCE_SHA256 = {
    "0-CMakeLists.txt": "ce855dc8624be405198ef5ffed9eb382323ad936a89b70d07851a69276ba0a4b",
    "1-CMakeLists.txt": "c01735e394f8f228e88c1f432c509b39785bf4f0b7d85cfa4ee36f106dd15d59",
    "2-pyproject.toml": "c973697610f77af63250643de693af1995a9fa1f3b5c221171b55d045aa0d4f5",
    "3-generate_torch_version.py": "cb708c746b2a25943accea069fa70f6e05d21bb17135521ad30bfb476241e254",
    "4-version.txt": "127662056a25294fe93b2d6c4c05c9f6e1556c55a1cfcd02fef79f33b08be2d6",
}
BUILD_IMAGE = "28ad36e62398875f0ad77112b1d0f2b1cd0b3b55560d9cc38c6908cada8caa06"
RUNTIME_IMAGE = "1bb4ac0ed3766d9c70431c1d5119e03cd4d10e710f4c04cd54254b5b87cd45d5"


def verify(package):
    record = json.loads((package / "evidence/result.json").read_text())
    assert record["status"] == "passed"
    for name, digest in record["sources"].items():
        assert sha(package / "tools" / name) == digest
    for name, digest in SOURCE_SHA256.items():
        assert sha(package / "source-proof" / name) == digest
    counts = {}
    for profile, name in [("daily", "b052"), ("rccl-single", "b053")]:
        saved = record["profiles"][profile]
        root = package / "evidence" / profile
        for path, digest in saved["files"].items():
            assert sha(root / path) == digest, path
        original = json.loads((root / "retained-build.json").read_text())
        assert saved["original"] == original and original["image"] == BUILD_IMAGE
        settings = build_settings((root / ("build_torch_" + name + ".sh")).read_text())
        assert settings == saved["settings"]
        assert settings["USE_NCCL"] == ("0" if profile == "daily" else "1")
        missing = saved["classification"]["missing"]
        classification = classify(
            (root / "CMakeCache.txt").read_text(),
            json.loads((root / "compile_commands.json").read_text()),
            missing,
        )
        assert classification == saved["classification"]
        assert classification["compile_entries"] == (
            3239 if profile == "daily" else 3241
        )
        ninja = (root / "build.ninja").read_text()
        assert all(term not in ninja for term in classification["references"])
        counts[profile] = classification["compile_entries"]
    for label, image in [("native-build", BUILD_IMAGE), ("runtime", RUNTIME_IMAGE)]:
        saved = record["images"][label]
        actual = json.loads((package / "evidence" / label / "result.json").read_text())
        assert (
            saved["image"] == image
            and actual == saved["result"]
            and actual["status"] == "passed"
        )
        assert not actual["torch_imported"] and not actual["gpu_nodes_exposed"]
        assert actual["cmake_on_path_sha256"] == actual["tools"]["cmake"]["sha256"]
        assert actual["tools"]["cmake"]["version"].startswith("cmake version 3.31.12")
        assert actual["build_packages"]["cmake"] == "4.4.0"
        assert all(
            len(tool["sha256"]) == 64 and tool["version"]
            for tool in actual["tools"].values()
        )
        command = saved["command"]
        assert (
            "--read-only" in command
            and "--network=none" in command
            and "--pull=never" in command
        )
        assert not any(x.startswith("--device") for x in command)
    assert (
        record["sdk_sha256"]
        == "2a975370ba4eb1c2fe39e406ff36b175f554a49d0aa19febbd326f641fcb590b"
    )
    assert all(
        s["available_bytes"] >= 3 * 1024**3 and s["disk_free_bytes"] >= 5 * 1024**3
        for s in record["samples"]
    )
    requirements = tomllib.loads(
        (package / "source-proof/2-pyproject.toml").read_text()
    )["build-system"]["requires"]
    return dict(
        status="passed",
        compile_entries=counts,
        missing_branches_dispositioned=8,
        required_missing_branch_fetches=0,
        native_build_image=BUILD_IMAGE,
        runtime_image=RUNTIME_IMAGE,
        original_build_packages=len(
            record["images"]["native-build"]["result"]["packages"]
        ),
        runtime_packages=len(record["images"]["runtime"]["result"]["packages"]),
        build_system_requirements=requirements,
        tool_version="CMake actual3.31.12 differs installed Python distribution4.4.0",
        manifest_sha256=sha(package / "evidence/result.json"),
        seconds=record["seconds"],
        min_host_available_bytes=min(x["available_bytes"] for x in record["samples"]),
        min_disk_free_bytes=min(x["disk_free_bytes"] for x in record["samples"]),
        limits="Native selected configuration only; image/wheel/toolchain payload export and full clean-build/license closure remain separate",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.package)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
