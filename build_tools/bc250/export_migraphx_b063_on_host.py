#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Preserve the strict C++17 repair and native reference evidence, without deployment."""
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile

from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS


def main():
    logs = WS / "logs"
    qualified = json.loads(
        (logs / "migraphx-b063-qualification-r2/final.json").read_text()
    )
    assert qualified["status"] == "passed"
    assert sum(g["checks"] for g in qualified["gates"]) == 93
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    for path, digest in qualified["protected_sha256"].items():
        assert sha(Path(path)) == digest, path
    assert (
        subprocess.check_output(["dmesg"])
        == (logs / "migraphx-b063-qualification-r2/kernel-before.log").read_bytes()
    )
    hashes = json.loads((WS / "migraphx-b063/source-sha256.json").read_text())
    for name, digest in hashes.items():
        assert sha(WS / "migraphx-b063/source" / name) == digest, name
    build = logs / "migraphx-b063-build-r4"
    built = json.loads((build / "final.json").read_text())
    assert built["status"] == "probes-built" and not built["changed_hashes"]
    assert int((build / "unsupported.exit").read_text()) != 0
    assert "subwave" in (build / "unsupported.log").read_text()
    features = json.loads((build / "features.json").read_text())
    assert features["status"] == "passed"
    for dpp in (0, 1):
        text = (build / f"device-dpp{dpp}.asm").read_text()
        assert text.strip() and not re.search(r"\bv_(?:mfma|wmma)\w*\b", text)
        assert "gfx1013" in (build / f"device-dpp{dpp}-notes.txt").read_text()
    for gate in qualified["gates"]:
        state = json.loads((Path("/run") / gate["label"] / "state.json").read_text())
        assert (
            state["status"] == "acknowledged"
            and state["boot_id"] == qualified["boot_id"]
        )
    fs = os.statvfs(WS)
    report = dict(
        status="qualified-header-repair",
        checks=93,
        gpu_gates=5,
        acknowledged_guards=5,
        source_files_verified=len(hashes),
        protected_paths=len(qualified["protected_sha256"]),
        boot_id=qualified["boot_id"],
        disk_budget=built["disk_budget"],
        ordinary_available_bytes=fs.f_bavail * fs.f_frsize,
        free_including_reserved_bytes=fs.f_bfree * fs.f_frsize,
        runtime_changed=False,
        full_backend_built=False,
        provider_installed=False,
        host105_contacted=False,
        reserve_policy_changed=False,
        running_jobs=[],
        features=features["values"]["gfx1013"],
    )
    dest = logs / "migraphx-b063-export"
    dest.mkdir()
    atomic_json(dest / "verification.json", report)
    archive = dest / "evidence.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(logs.glob("migraphx-b063-*")):
            if path != dest and "export-controller" not in path.name:
                tar.add(path, arcname="logs/" + path.name)
        for gate in qualified["gates"]:
            tar.add(Path("/run") / gate["label"], arcname="guards/" + gate["label"])
        for path in sorted((WS / "helpers").glob("*migraphx*b063*")):
            if path.is_file():
                tar.add(path, arcname="helpers/" + path.name)
        tar.add(WS / "migraphx-b063", arcname="sources")
        tar.add(WS / "tmp/migraphx-b063", arcname="builds")
        tar.add(dest / "verification.json", arcname="verification.json")
    atomic_json(
        dest / "archive.json", dict(sha256=sha(archive), bytes=archive.stat().st_size)
    )
    for path in dest.iterdir():
        os.chown(path, 1000, 1000)
    print("MIGRAPHX_B063_EXPORT_PASS", flush=True)


if __name__ == "__main__":
    main()
