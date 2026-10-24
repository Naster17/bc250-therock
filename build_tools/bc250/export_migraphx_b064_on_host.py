#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Seal the isolated graph evidence without deploying the candidate backend."""
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import tarfile

from build_migraphx_b064_on_host import STORAGE
from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS


def main():
    report_path = WS / "logs/migraphx-b064-qualification-r2/final.json"
    qualified = json.loads(report_path.read_text())
    assert (
        qualified["status"] == "passed"
        and sum(g["checks"] for g in qualified["gates"]) == 44
    )
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    for name, digest in qualified["protected_sha256"].items():
        assert sha(Path(name)) == digest, name
    before = (report_path.parent / "kernel-before.log").read_bytes()
    assert subprocess.check_output(["dmesg"]) == before
    for gate in qualified["gates"]:
        state = json.loads((Path("/run") / gate["label"] / "state.json").read_text())
        assert (
            state["status"] == "acknowledged"
            and state["boot_id"] == qualified["boot_id"]
        )
    # Verify staged native inputs against the exact archive backed by workstation disk.
    with tarfile.open(STORAGE / "source-r4.tar.gz") as tar:
        count = 0
        for member in tar.getmembers():
            if member.isfile():
                expected = hashlib.sha256(tar.extractfile(member).read()).hexdigest()
                assert (
                    sha(WS / "tmp/migraphx-b064/20261025-source-r4" / member.name)
                    == expected
                ), member.name
                count += 1

    with tarfile.open(WS / "migraphx-b062/rocm-cmake.tar.gz") as tar:
        dependency_count = 0
        for member in tar.getmembers():
            if member.isfile():
                expected = hashlib.sha256(tar.extractfile(member).read()).hexdigest()
                assert (
                    sha(WS / "migraphx-b062/rocm-cmake" / member.name) == expected
                ), member.name
                dependency_count += 1

    def bytes_free(path):
        fs = os.statvfs(path)
        return dict(
            ordinary_available_bytes=fs.f_bavail * fs.f_frsize,
            free_including_reserved_bytes=fs.f_bfree * fs.f_frsize,
        )

    sealed = dict(
        status="qualified-isolated-fp32-graphs",
        checks=44,
        gates=5,
        source_revision="622ea3f2ad8f1baa8a44edb4f1e07cbe6e532c83",
        source_files_verified=count,
        rocm_cmake_files_verified=dependency_count,
        protected_paths=len(qualified["protected_sha256"]),
        acknowledged_guards=5,
        boot_id=qualified["boot_id"],
        root_space=bytes_free(WS),
        durable_space=bytes_free(STORAGE),
        runtime_changed=False,
        provider_installed=False,
        inference_only=True,
        reserve_policy_changed=False,
        host105_state_changed=False,
        inherited_apk_105_mirror_request="Connection refused; no deployment or state change",
        running_gpu_jobs=[],
    )
    assembly = (STORAGE / "logs-fixture-r1/device.asm").read_text()
    notes = (STORAGE / "logs-fixture-r1/device-notes.txt").read_text()
    assert assembly.strip() and not re.search(r"\bv_(?:mfma|wmma)\w*\b", assembly)
    assert notes.count("amdhsa.target:") == 5 and "gfx1013" in notes
    sealed["native_device_objects"] = 5
    sealed["tested_device_objects_mfma_wmma"] = False
    dest = STORAGE / "export-r1"
    dest.mkdir()
    atomic_json(dest / "verification.json", sealed)
    archive = dest / "evidence.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted((WS / "logs").glob("*migraphx-b064*")):
            tar.add(path, arcname="host-logs/" + path.name)
        for path in sorted(STORAGE.glob("logs-*")):
            tar.add(path, arcname="build-logs/" + path.name)
        for gate in qualified["gates"]:
            tar.add(Path("/run") / gate["label"], arcname="guards/" + gate["label"])
        for path in sorted((WS / "helpers").glob("*migraphx*b064*")):
            if path.is_file():
                tar.add(path, arcname="helpers/" + path.name)
        for name in (
            "CMakeLists.txt",
            "source-identity.json",
            "source-identity-r4.json",
            "source-r4.tar.gz",
            "patch-r4.tar.gz",
            "source.tar.gz",
            "probe_migraphx_b064.cpp",
            "finish.sh",
        ):
            tar.add(STORAGE / name, arcname="inputs/" + name)
        for path in sorted(STORAGE.glob("syntax*.log")):
            tar.add(path, arcname="syntax/" + path.name)
        tar.add(STORAGE / "syntax.py", arcname="syntax/syntax.py")
        tar.add(STORAGE / "syntax-provider.py", arcname="syntax/syntax-provider.py")
        tar.add(STORAGE / "storage-contract", arcname="storage-contract")
        tar.add(STORAGE / "build-r1", arcname="build")
        tar.add(dest / "verification.json", arcname="verification.json")
    atomic_json(
        dest / "archive.json", dict(sha256=sha(archive), bytes=archive.stat().st_size)
    )
    print("MIGRAPHX_B064_EXPORT_PASS", flush=True)


if __name__ == "__main__":
    main()
