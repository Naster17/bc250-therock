#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export bounded module qualification, owned checkpoints and preservation evidence."""

import json
import os
from pathlib import Path
import subprocess
import tarfile

from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS

SOURCES = (
    "bc250_quant_b051.py",
    "validate_quant_b051.py",
    "bc250_quant_b055.py",
    "validate_quant_b055.py",
    "qualify_quant_b055_on_host.py",
    "export_quant_b055_on_host.py",
)


def main():
    assert os.geteuid() == 0
    logs = WS / "logs/quant-b055-qualification-r1"
    result = json.loads((logs / "final.json").read_text())
    assert result["status"] == "passed"
    assert len(result["gates"]) == 10
    assert sum(g["checks"] for g in result["gates"]) == 295
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    assert (
        subprocess.check_output(["dmesg"]) == (logs / "kernel-before.log").read_bytes()
    )
    assert (
        Path("/proc/sys/kernel/random/boot_id").read_text().strip() == result["boot_id"]
    )
    for name, digest in result["protected_sha256"].items():
        assert sha(Path(name)) == digest, name
    for name, target in result["links"].items():
        assert os.readlink(name) == target, name
    for g in result["gates"]:
        state = json.loads((Path("/run") / g["label"] / "state.json").read_text())
        assert (
            state["status"] == "acknowledged" and state["boot_id"] == result["boot_id"]
        )
    destination = WS / "logs/quant-b055-export"
    destination.mkdir()
    report = dict(
        status="passed",
        checks=295,
        gates=10,
        protected_paths=len(result["protected_sha256"]),
        source_sha256={name: sha(WS / "helpers" / name) for name in SOURCES},
        daily_runtime_sha256=sha(Path("/opt/bc250-stack/current/runtime.json")),
        profile_sha256=sha(Path("/opt/bc250-stack/profiles/rccl-single/runtime.json")),
        boot_id=result["boot_id"],
        acknowledged_guards=10,
        native_prefix_changed=False,
        native_build_required=False,
        host105_contacted=False,
        running_jobs=[],
    )
    atomic_json(destination / "verification.json", report)
    archive = destination / "evidence.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for g in result["gates"]:
            label = g["label"]
            tar.add(WS / "logs" / label, arcname="logs/" + label)
            tar.add(Path("/run") / label, arcname="guards/" + label)
        tar.add(logs, arcname="logs/" + logs.name)
        for name in SOURCES:
            tar.add(WS / "helpers" / name, arcname="helpers/" + name)
        tar.add(destination / "verification.json", arcname="verification.json")
    atomic_json(
        destination / "archive.json",
        dict(sha256=sha(archive), bytes=archive.stat().st_size),
    )
    for p in destination.iterdir():
        os.chown(p, 1000, 1000)
    print("QUANT_B055_EXPORT_PASS", flush=True)


if __name__ == "__main__":
    main()
