#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Seal isolated parser qualification without installing it into the active runtime."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

from build_migraphx_b066_on_host import STORAGE
from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS


def main():
    logs = WS / "logs"
    qualified = json.loads(
        (logs / "migraphx-b066-qualification-r2/final.json").read_text()
    )
    assert (
        qualified["status"] == "passed"
        and sum(g["checks"] for g in qualified["gates"]) == 169
    )
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    for name, digest in qualified["protected_sha256"].items():
        assert sha(Path(name)) == digest, name
    assert (
        subprocess.check_output(["dmesg"])
        == (logs / "migraphx-b066-qualification-r2/kernel-before.log").read_bytes()
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
    metadata = json.loads((STORAGE / "pybind-source.json").read_text())
    for filename, digest in metadata["source_sha256"].items():
        assert sha(WS / "tmp/migraphx-b066/pybind11" / filename) == digest, filename
    deps = dict(
        pybind11=dict(
            revision=metadata["commit"],
            source_files_verified=len(metadata["source_sha256"]),
        )
    )

    def free(path):
        fs = os.statvfs(path)
        return dict(
            ordinary_available_bytes=fs.f_bavail * fs.f_frsize,
            free_including_reserved_bytes=fs.f_bfree * fs.f_frsize,
        )

    report = dict(
        status="qualified-isolated-python-onnx-fp32",
        checks=169,
        gates=7,
        acknowledged_guards=7,
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
        python_bindings=True,
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
        for path in sorted(logs.glob("*migraphx-b066*")):
            tar.add(path, arcname="host-logs/" + path.name)
        for path in sorted(STORAGE.glob("logs-*")):
            tar.add(path, arcname="build-logs/" + path.name)
        for gate in qualified["gates"]:
            tar.add(Path("/run") / gate["label"], arcname="guards/" + gate["label"])
        for path in sorted((WS / "helpers").glob("*migraphx*b066*")):
            if path.is_file():
                tar.add(path, arcname="helpers/" + path.name)
        for name in (
            "CMakeLists.txt",
            "pybind-source.json",
            "probe_migraphx_b064.cpp",
            "probe_migraphx_b065.cpp",
        ):
            tar.add(STORAGE / name, arcname="inputs/" + name)
        tar.add(
            STORAGE.parent / "b065/models-generation-r2.json",
            arcname="inputs/models-generation-r2.json",
        )
        tar.add(STORAGE.parent / "b065/models-r2", arcname="models")
        for subdir in ("lib", "bin"):
            tar.add(
                STORAGE / "build-r1/migraphx" / subdir, arcname="candidate/" + subdir
            )
        for name in ("build-r1", "build-pybind"):
            for filename in ("CMakeCache.txt", "build.ninja", ".ninja_log"):
                path = STORAGE / name / filename
                if path.exists():
                    tar.add(path, arcname=name + "/" + filename)
        tar.add(dest / "verification.json", arcname="verification.json")
    atomic_json(
        dest / "archive.json", dict(sha256=sha(archive), bytes=archive.stat().st_size)
    )
    print("MIGRAPHX_B065_EXPORT_PASS", flush=True)


if __name__ == "__main__":
    main()
