#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Seal the bounded CK plugin admission audit and isolated factory evidence."""
import json
import os
from pathlib import Path
import subprocess
import tarfile

from bc250_torch import runtime_lock
from build_migraphx_b067_on_host import verify_hashes
from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS


def main():
    logs = WS / "logs/ck-plugin-b071-qualification-r2"
    a = json.loads((logs / "final.json").read_text())
    assert a["status"] == "passed" and sum(x["checks"] for x in a["gates"]) == 24
    dest = WS / "tmp/ck-plugin-b071/export-r1"
    dest.mkdir()
    with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
        assert not verify_hashes(a["protected_sha256"])
        assert not verify_hashes(a["binary_sha256"])
        assert all(os.readlink(p) == v for p, v in a["links"].items())
        assert (
            subprocess.check_output(["dmesg"])
            == (logs / "kernel-before.log").read_bytes()
        )
        assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
        guards = []
        for path in Path("/run").glob("ck-plugin-b071-*/state.json"):
            s = json.loads(path.read_text())
            assert s["status"] == "acknowledged" and s["boot_id"] == a["boot_id"]
            guards.append(path.parent)
        assert len(guards) == 6
        probes = {}
        for cohort in ("daily", "rccl"):
            r = json.loads(
                (WS / f"logs/ck-plugin-b071-r2-{cohort}/result.json").read_text()
            )
            assert r["status"] == "passed"
            probes[cohort] = r["probes"]
        r = dict(
            status="qualified-bounded-plugin-admission-factory",
            checks=24,
            accepted_guards=5,
            total_acknowledged_guards=6,
            probes=probes,
            protected_paths=len(a["protected_sha256"]),
            binary_sha256=a["binary_sha256"],
            links=len(a["links"]),
            aot_caches=a["aot_caches"],
            boot_id=a["boot_id"],
            kernel=os.uname().release,
            kernel_log_unchanged=True,
            active_runtime_changed=False,
            plugin_built=False,
            plugin_installed=False,
            host105_contacted=False,
            running_owned_jobs=[],
            ordinary_available_bytes=os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize,
        )
        atomic_json(dest / "verification.json", r)
        audit = json.loads((WS / "logs/ck-plugin-b071-source-audit.json").read_text())
        for n, h in audit["source_sha256"].items():
            assert sha(WS / "source/rocm-libraries" / n) == h
        with tarfile.open(dest / "evidence.tar.gz", "w:gz") as tar:
            for path in sorted((WS / "logs").glob("*ck-plugin-b071*")):
                tar.add(path, arcname="host-logs/" + path.name)
            for path in guards:
                tar.add(path, arcname="guards/" + path.name)
            for path in sorted((WS / "helpers").glob("*b071*")):
                if path.is_file():
                    tar.add(path, arcname="helpers/" + path.name)
            for name in ("build-r1", "build-r2"):
                tar.add(WS / "tmp/ck-plugin-b071" / name, arcname=name)
            for n in audit["source_sha256"]:
                tar.add(WS / "source/rocm-libraries" / n, arcname="sources/" + n)
        atomic_json(
            dest / "archive.json",
            dict(
                sha256=sha(dest / "evidence.tar.gz"),
                bytes=(dest / "evidence.tar.gz").stat().st_size,
            ),
        )
    print("CK_PLUGIN_B071_EVIDENCE_PASS", flush=True)


if __name__ == "__main__":
    main()
