#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export source/configure/header evidence without promoting a runtime."""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

from deploy_torch_a04_on_host import sha, atomic_json
from qualify_training_n33_on_host import WS

SOURCES = (
    "probe_migraphx_b062.hip",
    "build_migraphx_b062.sh",
    "build_migraphx_b062_on_host.py",
    "configure_migraphx_b062_on_host.py",
    "validate_migraphx_b062.py",
    "qualify_migraphx_b062_on_host.py",
    "audit_migraphx_b062.py",
    "export_migraphx_b062_on_host.py",
)


def main():
    qualified = json.loads(
        (WS / "logs/migraphx-b062-qualification-r1/final.json").read_text()
    )
    assert (
        qualified["status"] == "passed"
        and sum(g["checks"] for g in qualified["gates"]) == 52
    )
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    for name, digest in qualified["protected_sha256"].items():
        assert sha(Path(name)) == digest, name
    assert (
        subprocess.check_output(["dmesg"])
        == (WS / "logs/migraphx-b062-probe-r1/kernel-before.log").read_bytes()
    )
    source = WS / "migraphx-b062/source"
    audit = json.loads((WS / "migraphx-b062/source-audit.json").read_text())
    for name, digest in audit["source_sha256"].items():
        assert sha(source / name) == digest, name
    config = WS / "logs/migraphx-b062-configure-r1"
    assert int((config / "configure-exit.txt").read_text()) == 0
    assert (config / "strict-dpp1.exit").read_text().strip() == "0"
    assert (config / "strict-dpp0.exit").read_text().strip() == "1"
    assert "C++20 extension" in (config / "strict-dpp0.log").read_text()
    assembly = WS / "logs/migraphx-b062-build-r2"
    for dpp in (0, 1):
        data = (assembly / f"device-dpp{dpp}.asm").read_text()
        assert data.strip() and not re.search(r"\bv_(?:mfma|wmma)\w*\b", data)
        assert "gfx1013" in (assembly / f"device-dpp{dpp}-notes.txt").read_text()
    for g in qualified["gates"]:
        guard = json.loads((Path("/run") / g["label"] / "state.json").read_text())
        assert (
            guard["status"] == "acknowledged"
            and guard["boot_id"] == qualified["boot_id"]
        )
    dest = WS / "logs/migraphx-b062-export"
    dest.mkdir()
    report = dict(
        status="audit-qualified-header-scope",
        checks=52,
        gates=5,
        acknowledged_guards=5,
        protected_paths=len(qualified["protected_sha256"]),
        source_revision=audit["revision"],
        rocm_cmake_revision=audit["rocm_cmake_revision"],
        source_files_verified=len(audit["source_sha256"]),
        source_unmodified=True,
        source_sha256={n: sha(WS / "helpers" / n) for n in SOURCES},
        daily_runtime_sha256=sha(Path("/opt/bc250-stack/current/runtime.json")),
        profile_sha256=sha(Path("/opt/bc250-stack/profiles/rccl-single/runtime.json")),
        boot_id=qualified["boot_id"],
        minimal_gpu_configure=True,
        strict_dpp_default=0,
        strict_dpp_disabled=1,
        full_backend_built=False,
        provider_installed=False,
        runtime_changed=False,
        free_bytes=shutil.disk_usage(WS).free,
        host105_contacted=False,
        running_jobs=[],
    )
    atomic_json(dest / "verification.json", report)
    archive = dest / "evidence.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for p in sorted((WS / "logs").glob("migraphx-b062-*")):
            if p != dest and "export-controller" not in p.name:
                tar.add(p, arcname="logs/" + p.name)
        for g in qualified["gates"]:
            tar.add(Path("/run") / g["label"], arcname="guards/" + g["label"])
        for name in SOURCES:
            tar.add(WS / "helpers" / name, arcname="helpers/" + name)
        tar.add(WS / "migraphx-b062/source.tar.gz", arcname="sources/migraphx.tar.gz")
        tar.add(
            WS / "migraphx-b062/rocm-cmake.tar.gz", arcname="sources/rocm-cmake.tar.gz"
        )
        tar.add(WS / "migraphx-b062/source-audit.json", arcname="source-audit.json")
        tar.add(WS / "tmp/migraphx-b062/20261024-r2/tmp", arcname="probe-build")
        tar.add(WS / "tmp/migraphx-b062/20261024-configure", arcname="configure-build")
        tar.add(dest / "verification.json", arcname="verification.json")
    atomic_json(
        dest / "archive.json", dict(sha256=sha(archive), bytes=archive.stat().st_size)
    )
    for p in dest.iterdir():
        os.chown(p, 1000, 1000)
    print("MIGRAPHX_B062_EXPORT_PASS", flush=True)


if __name__ == "__main__":
    main()
