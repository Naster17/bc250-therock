#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export W8 wheel, package/runtime qualification and guarded rollback evidence."""

import json
import os
from pathlib import Path
import subprocess
import tarfile

from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS
from stage_quant_b056_on_host import ACTIVE

SOURCES = (
    "build_quant_b056.py",
    "stage_quant_b056_on_host.py",
    "qualify_quant_b056_on_host.py",
    "validate_quant_b056.py",
    "validate_quant_b056_compat.py",
    "deploy_quant_b056_on_host.py",
    "export_quant_b056_on_host.py",
    "bc250_quant_b051.py",
    "bc250_quant_b055.py",
)


def main():
    assert os.geteuid() == 0
    stage = json.loads((WS / "logs/quant-b056-stage-r3/final.json").read_text())
    deployed = json.loads((WS / "logs/quant-b056-deployment.json").read_text())
    assert deployed["status"] == "passed"
    reports = {
        phase: json.loads(
            (WS / ("logs/quant-b056-" + phase + "-r3/final.json")).read_text()
        )
        for phase in ("candidate", "installed")
    }
    assert all(
        r["status"] == "passed" and sum(g["checks"] for g in r["gates"]) == 808
        for r in reports.values()
    )
    installed = reports["installed"]
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    for name, digest in installed["protected_sha256"].items():
        assert sha(Path(name)) == digest, name
    assert all(os.readlink(p) == v for p, v in installed["links"].items())
    assert (
        subprocess.check_output(["dmesg"])
        == (WS / "logs/quant-b056-candidate-r3/kernel-before.log").read_bytes()
    )
    guards = [g["label"] for r in reports.values() for g in r["gates"]]
    for phase in ("candidate", "installed"):
        label = (
            "quant-b056-"
            + phase
            + ("-compat-r1" if phase == "candidate" else "-compat-r3")
        )
        folder = WS / "logs" / label
        result = json.loads((folder / "result.json").read_text())
        assert result["status"] == "passed" and len(result["checks"]) == 9
        exit_state = json.loads((folder / "exit.json").read_text())
        assert exit_state["returncode"] == 0 and exit_state["kernel_log_unchanged"]
        guards.append(label)
    assert len(guards) == 32
    accepted_guards = list(guards)
    historical = []
    for pattern in (
        "quant-b056-candidate-r1-*",
        "quant-b056-installed-r1-*",
        "quant-b056-candidate-r2-*",
        "quant-b056-installed-r2-*",
    ):
        historical.extend(p.name for p in (WS / "logs").glob(pattern) if p.is_dir())
    assert len(historical) == 49
    guards.extend(historical)
    for label in guards:
        guard = json.loads((Path("/run") / label / "state.json").read_text())
        assert (
            guard["status"] == "acknowledged"
            and guard["boot_id"] == installed["boot_id"]
        )
    destination = WS / "logs/quant-b056-export"
    destination.mkdir()
    images = json.loads(
        subprocess.check_output(
            ["podman", "image", "inspect", stage["previous_image"], stage["image"]],
            text=True,
        )
    )
    atomic_json(destination / "images.json", images)
    report = dict(
        status="passed",
        candidate_checks=817,
        installed_checks=817,
        gates=32,
        protected_paths=len(installed["protected_sha256"]),
        source_sha256={n: sha(WS / "helpers" / n) for n in SOURCES},
        daily_runtime_sha256=sha(ACTIVE["daily"] / "runtime.json"),
        profile_sha256=sha(ACTIVE["rccl"] / "runtime.json"),
        image=stage["image"],
        previous_image=stage["previous_image"],
        base_layers_preserved=stage["base_layers_preserved"],
        wheel_sha256=stage["wheel_sha256"],
        wheel_bytes=Path(stage["wheel"]).stat().st_size,
        recovery=stage["recovery"],
        boot_id=installed["boot_id"],
        acknowledged_guards=len(guards),
        accepted_guards=len(accepted_guards),
        historical_guards=len(historical),
        rollback_exercised=True,
        native_libraries_changed=False,
        host105_contacted=False,
        running_jobs=[],
        rollback_used=deployed["rollback_used"],
    )
    atomic_json(destination / "verification.json", report)
    archive = destination / "evidence.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for label in guards:
            tar.add(WS / "logs" / label, arcname="logs/" + label)
            tar.add(Path("/run") / label, arcname="guards/" + label)
        for name in (
            "quant-b056-candidate-r1",
            "quant-b056-installed-r1",
            "quant-b056-candidate-r2",
            "quant-b056-installed-r2",
        ):
            tar.add(WS / "logs" / name, arcname="logs/" + name)
        for phase in ("candidate", "installed"):
            name = "quant-b056-" + phase + "-r3"
            tar.add(WS / "logs" / name, arcname="logs/" + name)
        for p in sorted((WS / "logs").glob("quant-b056-stage-*")):
            tar.add(p, arcname="logs/" + p.name)
        tar.add(
            WS / "logs/quant-b056-stage-controller.log",
            arcname="logs/quant-b056-stage-controller.log",
        )
        tar.add(
            WS / "logs/quant-b056-deployment.json",
            arcname="logs/quant-b056-deployment.json",
        )
        tar.add(stage["wheel"], arcname="wheel/" + Path(stage["wheel"]).name)
        tar.add(
            Path(stage["wheel"]).parent / "Containerfile", arcname="wheel/Containerfile"
        )
        tar.add(Path(stage["recovery"]), arcname="recovery")
        for n in SOURCES:
            tar.add(WS / "helpers" / n, arcname="helpers/" + n)
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
    print("QUANT_B056_EXPORT_PASS", flush=True)


if __name__ == "__main__":
    main()
