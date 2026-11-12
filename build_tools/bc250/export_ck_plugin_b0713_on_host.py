#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Seal B07.13 installed explicit activation, refusal/rollback and plain-default evidence."""
import json
import os
from pathlib import Path
import subprocess
import tarfile
import bc250_ck_b0713 as activation
from bc250_torch import LaunchOptions, runtime_lock
from build_migraphx_b067_on_host import verify_hashes
from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS

GATE_CHECKS = 72


def main():
    logs = WS / "logs/ck-plugin-b0713-qualification-r1"
    a = json.loads((logs / "final.json").read_text())
    assert (
        a["status"] == "passed" and sum(g["checks"] for g in a["gates"]) == GATE_CHECKS
    )
    user = json.loads((WS / "logs/ck-plugin-b0713-user-r1/final.json").read_text())
    assert user["status"] == "passed" and user["checks"] == 2
    out = WS / "tmp/ck-plugin-b0713/export-r1"
    out.mkdir()
    with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
        assert not verify_hashes(a["protected_sha256"])
        assert all(os.readlink(p) == v for p, v in a["links"].items())
        assert (
            subprocess.check_output(["dmesg"])
            == (logs / "kernel-before.log").read_bytes()
        )
        assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
        record, root = activation.registered(LaunchOptions())
        activation.registered(LaunchOptions(runtime_profile="rccl-single"))
        guards = []
        for p in Path("/run").glob("ck-plugin-b0713-*/state.json"):
            state = json.loads(p.read_text())
            assert (
                state["status"] == "acknowledged" and state["boot_id"] == a["boot_id"]
            )
            guards.append(p.parent)
        assert len(guards) == 10
        cohorts = {
            name: json.loads(
                (WS / f"logs/ck-plugin-b0713-r1-{name}/result.json").read_text()
            )
            for name in (
                "installed-daily",
                "installed-rccl",
                "plain-daily",
                "plain-rccl",
                "host-cli",
            )
        }
        report = dict(
            status="qualified-installed-explicit-option",
            guarded_checks=GATE_CHECKS + 2,
            cpu_checks=len(a["cpu_checks"]),
            total_checks=GATE_CHECKS + 2 + len(a["cpu_checks"]),
            accepted_guards=len(guards),
            registration=record,
            cohorts=cohorts,
            user=user,
            protected_paths=len(a["protected_sha256"]),
            links=len(a["links"]),
            boot_id=a["boot_id"],
            kernel=os.uname().release,
            native_runtime_changed=False,
            default_admission_changed=False,
            host105_contacted=False,
            running_owned_jobs=[],
            ordinary_free_bytes=os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize,
        )
        atomic_json(out / "verification.json", report)
        with tarfile.open(out / "evidence.tar.gz", "w:gz") as tar:

            def retain(info):
                return None if "bad-package" in info.name else info

            for p in sorted((WS / "logs").glob("*ck-plugin-b0713*")):
                tar.add(p, arcname="host-logs/" + p.name, filter=retain)
            for p in guards:
                tar.add(p, arcname="guards/" + p.name)
            for p in (WS / "helpers").glob("*b0713*"):
                if p.is_file():
                    tar.add(p, arcname="helpers/" + p.name)
            for p in Path("/opt/bc250-stack/tools/ck-b0713").iterdir():
                if p.is_file():
                    tar.add(p, arcname="installed-tools/" + p.name)
            tar.add(activation.REGISTRY, arcname="installed/miopen-dl.json")
            tar.add(root / "manifest.json", arcname="installed/manifest.json")
            tar.add(
                activation.BASE / "miopen-dl-b0713-rollback.json",
                arcname="installed/miopen-dl-b0713-rollback.json",
            )
        atomic_json(
            out / "archive.json",
            dict(
                sha256=sha(out / "evidence.tar.gz"),
                bytes=(out / "evidence.tar.gz").stat().st_size,
            ),
        )
    print("CK_PLUGIN_B0713_EVIDENCE_PASS", flush=True)


if __name__ == "__main__":
    main()
