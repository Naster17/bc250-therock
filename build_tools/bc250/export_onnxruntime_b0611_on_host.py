#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Seal installed ORT package, relocation, recovery and preserved-runtime evidence."""
import json
import os
from pathlib import Path
import subprocess
import tarfile

from bc250_ort import registered, REGISTRY, NAME, BASE as INSTALLED
from bc250_torch import LaunchOptions, runtime_lock
from build_migraphx_b067_on_host import verify_hashes
from deploy_torch_a04_on_host import atomic_json, sha
from package_onnxruntime_b0611_on_host import BASE
from qualify_training_n33_on_host import WS


def main():
    a = json.loads((WS / "logs/onnxruntime-b0611-acceptance-r2/final.json").read_text())
    q = json.loads((WS / "logs/onnxruntime-b0611-relocation-r1/final.json").read_text())
    assert a["status"] == q["status"] == "passed"
    assert sum(g["checks"] for g in a["gates"]) == 90 and len(a["cpu_checks"]) == 14
    assert sum(g["checks"] for g in q["gates"]) == 68
    dest = BASE / "export-r1"
    dest.mkdir()
    packaged = json.loads((BASE / "package-r1/final.json").read_text())
    with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
        record, root = registered(LaunchOptions())
        registered(LaunchOptions(runtime_profile="rccl-single"))
        assert not verify_hashes(a["protected_sha256"])
        assert all(os.readlink(p) == v for p, v in a["links"].items())
        assert sha(Path(packaged["archive"])) == packaged["archive_sha256"]
        assert (
            subprocess.check_output(["dmesg"])
            == (
                WS / "logs/onnxruntime-b0611-acceptance-r2/kernel-before.log"
            ).read_bytes()
        )
        assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
        user = json.loads((WS / "logs/onnxruntime-b0611-user-info.json").read_text())
        old = json.loads(
            (WS / "recovery/ort-b0611-before-context/ort-migraphx.json").read_text()
        )
        assert user == old
        assert (
            WS / "logs/onnxruntime-b0611-context-after.log"
        ).read_text().strip() == "CALLER_CONTEXT_PASS"
        guards = {}
        for path in Path("/run").glob("onnxruntime-b0611-*"):
            if not (path / "state.json").exists():
                continue
            state = json.loads((path / "state.json").read_text())
            assert (
                state["status"] == "acknowledged" and state["boot_id"] == a["boot_id"]
            )
            guards[path.name] = path
        assert len(guards) == 14
        report = dict(
            status="qualified-installed-explicit-ort-package",
            checks=174,
            guarded_checks=158,
            cpu_registration_checks=14,
            normal_user_activation_checks=2,
            accepted_guards=8,
            total_acknowledged_guards=14,
            gpu_graph_evaluations=48,
            earlier_installed_gpu_evaluations=24,
            registration=record,
            package=packaged,
            installed=True,
            active_runtime_changed=False,
            native_protected_paths=len(a["protected_sha256"]),
            links=len(a["links"]),
            aot_caches=a["aot_caches"],
            boot_id=a["boot_id"],
            kernel=os.uname().release,
            kernel_log_unchanged=True,
            ordinary_available_bytes=os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize,
            host105_contacted=False,
            running_owned_jobs=[],
            next_action="B06.12 bounded end-to-end Torch exporter to public ORT GPU qualification and practical gap review",
        )
        atomic_json(dest / "verification.json", report)
        with tarfile.open(dest / "evidence.tar.gz", "w:gz") as tar:
            for path in sorted((WS / "logs").glob("*onnxruntime-b0611*")):
                tar.add(path, arcname="host-logs/" + path.name)
            for name, path in guards.items():
                tar.add(path, arcname="guards/" + name)
            for path in sorted((WS / "helpers").glob("*b0611*")):
                if path.is_file():
                    tar.add(path, arcname="helpers/" + path.name)
            tar.add(WS / "helpers/bc250_ort.py", arcname="helpers/bc250_ort.py")
            for path in (BASE / "package-r1").iterdir():
                if path.is_file() and not path.name.endswith(".tar.gz"):
                    tar.add(path, arcname="packaging/" + path.name)
            for path in Path("/opt/bc250-stack/tools/ort-b0611").glob("*.py"):
                tar.add(path, arcname="installed-tools/" + path.name)
            tar.add(REGISTRY, arcname="registration/active.json")
            tar.add(INSTALLED / (NAME + ".json"), arcname="registration/versioned.json")
            tar.add(
                WS / "recovery/ort-b0611-before-context",
                arcname="recovery/before-context",
            )
        atomic_json(
            dest / "archive.json",
            dict(
                sha256=sha(dest / "evidence.tar.gz"),
                bytes=(dest / "evidence.tar.gz").stat().st_size,
            ),
        )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
