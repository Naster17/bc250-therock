#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Seal isolated parser qualification without installing it into the active runtime."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

from build_migraphx_b065_on_host import STORAGE
from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS


def main():
    logs = WS / "logs"
    qualified = json.loads(
        (logs / "migraphx-b065-qualification-r1/final.json").read_text()
    )
    assert (
        qualified["status"] == "passed"
        and sum(g["checks"] for g in qualified["gates"]) == 65
    )
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    for name, digest in qualified["protected_sha256"].items():
        assert sha(Path(name)) == digest, name
    assert (
        subprocess.check_output(["dmesg"])
        == (logs / "migraphx-b065-qualification-r1/kernel-before.log").read_bytes()
    )
    for gate in qualified["gates"]:
        state = json.loads((Path("/run") / gate["label"] / "state.json").read_text())
        assert (
            state["status"] == "acknowledged"
            and state["boot_id"] == qualified["boot_id"]
        )
    with tarfile.open(STORAGE.parent / "source-r4.tar.gz") as tar:
        source_count = 0
        for member in tar.getmembers():
            if member.isfile():
                expected = hashlib.sha256(tar.extractfile(member).read()).hexdigest()
                assert (
                    sha(WS / "tmp/migraphx-b064/20261025-source-r4" / member.name)
                    == expected
                ), member.name
                source_count += 1
    deps = {}
    for name in ("abseil", "protobuf"):
        metadata = json.loads((STORAGE / (name + "-source.json")).read_text())
        assert sha(STORAGE / (name + ".tar.gz")) == metadata["archive_sha256"]
        for filename, digest in metadata["source_sha256"].items():
            assert sha(WS / "tmp/migraphx-b065" / name / filename) == digest, filename
        deps[name] = dict(
            revision=metadata["revision"],
            source_files_verified=len(metadata["source_sha256"]),
        )

    def free(path):
        fs = os.statvfs(path)
        return dict(
            ordinary_available_bytes=fs.f_bavail * fs.f_frsize,
            free_including_reserved_bytes=fs.f_bfree * fs.f_frsize,
        )

    report = dict(
        status="qualified-isolated-onnx-fp32-parser",
        checks=65,
        gates=6,
        acknowledged_guards=6,
        source_revision="622ea3f2ad8f1baa8a44edb4f1e07cbe6e532c83",
        source_files_verified=source_count,
        dependencies=deps,
        protected_paths=len(qualified["protected_sha256"]),
        boot_id=qualified["boot_id"],
        root_space=free(WS),
        durable_space=free(STORAGE),
        runtime_changed=False,
        provider_installed=False,
        onnx_parser_built=True,
        python_bindings=False,
        ort_gpu_provider=False,
        inference_only=True,
        reserve_policy_changed=False,
        host105_contacted=False,
        running_gpu_jobs=[],
    )
    dest = STORAGE / "export-r1"
    dest.mkdir()
    atomic_json(dest / "verification.json", report)
    archive = dest / "evidence.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(logs.glob("*migraphx-b065*")):
            tar.add(path, arcname="host-logs/" + path.name)
        for path in sorted(STORAGE.glob("logs-*")):
            tar.add(path, arcname="build-logs/" + path.name)
        for gate in qualified["gates"]:
            tar.add(Path("/run") / gate["label"], arcname="guards/" + gate["label"])
        for path in sorted((WS / "helpers").glob("*migraphx*b065*")):
            if path.is_file():
                tar.add(path, arcname="helpers/" + path.name)
        for name in (
            "CMakeLists.txt",
            "abseil-source.json",
            "protobuf-source.json",
            "abseil.tar.gz",
            "protobuf.tar.gz",
            "probe_migraphx_b064.cpp",
            "probe_migraphx_b065.cpp",
            "models-generation.json",
        ):
            tar.add(STORAGE / name, arcname="inputs/" + name)
        tar.add(STORAGE / "models", arcname="models")
        tar.add(STORAGE / "build-r1", arcname="parser-build")
        for name in ("build-abseil", "build-protobuf"):
            for filename in ("CMakeCache.txt", "build.ninja", ".ninja_log"):
                tar.add(STORAGE / name / filename, arcname=name + "/" + filename)
        tar.add(dest / "verification.json", arcname="verification.json")
    atomic_json(
        dest / "archive.json", dict(sha256=sha(archive), bytes=archive.stat().st_size)
    )
    print("MIGRAPHX_B065_EXPORT_PASS", flush=True)


if __name__ == "__main__":
    main()
