#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Archive ONNX CPU/export qualification, pinned wheels and runtime recovery."""

import json
import os
from pathlib import Path
import subprocess
import tarfile

from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS
from stage_quant_b056_on_host import ACTIVE

SOURCES = (
    "fetch_onnx_b061.py",
    "stage_onnx_b061_on_host.py",
    "qualify_onnx_b061_on_host.py",
    "deploy_onnx_b061_on_host.py",
    "export_onnx_b061_on_host.py",
    "validate_onnx_b061.py",
    "onnx_b061_packages.json",
)


def main():
    assert os.geteuid() == 0
    stage = json.loads((WS / "logs/onnx-b061-stage-r1/final.json").read_text())
    deployed = json.loads((WS / "logs/onnx-b061-deployment.json").read_text())
    assert deployed["status"] == "passed"
    reports = {
        phase: json.loads(
            (WS / ("logs/onnx-b061-" + phase + "-r1/final.json")).read_text()
        )
        for phase in ("candidate", "installed")
    }
    assert all(
        r["status"] == "passed" and sum(g["checks"] for g in r["gates"]) == 181
        for r in reports.values()
    )
    installed = reports["installed"]
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    for name, digest in installed["protected_sha256"].items():
        assert sha(Path(name)) == digest, name
    assert all(os.readlink(p) == v for p, v in installed["links"].items())
    assert (
        subprocess.check_output(["dmesg"])
        == (WS / "logs/onnx-b061-candidate-r1/kernel-before.log").read_bytes()
    )
    guards = [g["label"] for r in reports.values() for g in r["gates"]] + [
        "onnx-b061-probe-r1"
    ]
    assert len(guards) == 23
    for label in guards:
        guard = json.loads((Path("/run") / label / "state.json").read_text())
        assert (
            guard["status"] == "acknowledged"
            and guard["boot_id"] == installed["boot_id"]
        )
    destination = WS / "logs/onnx-b061-export"
    destination.mkdir()
    atomic_json(
        destination / "images.json",
        json.loads(
            subprocess.check_output(
                ["podman", "image", "inspect", stage["previous_image"], stage["image"]],
                text=True,
            )
        ),
    )
    report = dict(
        status="passed",
        candidate_checks=181,
        installed_checks=181,
        early_probe_checks=33,
        gates=22,
        protected_paths=len(installed["protected_sha256"]),
        source_sha256={n: sha(WS / "helpers" / n) for n in SOURCES},
        daily_runtime_sha256=sha(ACTIVE["daily"] / "runtime.json"),
        profile_sha256=sha(ACTIVE["rccl"] / "runtime.json"),
        image=stage["image"],
        previous_image=stage["previous_image"],
        base_layers_preserved=stage["base_layers_preserved"],
        packages=stage["packages"],
        recovery=stage["recovery"],
        boot_id=installed["boot_id"],
        acknowledged_guards=23,
        native_libraries_changed=False,
        host105_contacted=False,
        running_jobs=[],
        rollback_used=deployed["rollback_used"],
        scope="Modern ONNX export and explicit CPUExecutionProvider inference; host transfer to/from native Torch GPU, no ONNX GPU provider or ONNX autograd",
    )
    atomic_json(destination / "verification.json", report)
    archive = destination / "evidence.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for label in guards:
            tar.add(WS / "logs" / label, arcname="logs/" + label)
            tar.add(Path("/run") / label, arcname="guards/" + label)
        for phase in ("candidate", "installed"):
            name = "onnx-b061-" + phase + "-r1"
            tar.add(WS / "logs" / name, arcname="logs/" + name)
        for name in (
            "onnx-b061-stage-r1",
            "onnx-b061-deployment.json",
            "onnx-b061-qualification-start-refusal.log",
        ):
            tar.add(WS / "logs" / name, arcname="logs/" + name)
        tar.add(stage["context"], arcname="context")
        tar.add(stage["recovery"], arcname="recovery")
        tar.add(WS / "helpers/b061-probe", arcname="probe-source")
        for name in SOURCES:
            tar.add(WS / "helpers" / name, arcname="helpers/" + name)
        for name, active in ACTIVE.items():
            tar.add(active / "runtime.json", arcname="runtime/" + name + ".json")
        for n in ("verification.json", "images.json"):
            tar.add(destination / n, arcname=n)
    atomic_json(
        destination / "archive.json",
        dict(sha256=sha(archive), bytes=archive.stat().st_size),
    )
    for p in destination.iterdir():
        os.chown(p, 1000, 1000)
    print("ONNX_B061_EXPORT_PASS", flush=True)


if __name__ == "__main__":
    main()
