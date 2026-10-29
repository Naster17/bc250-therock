#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Seal per-ID timing results and their unchanged native-runtime baseline."""
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
    logs = WS / "logs/ck-plugin-b076-qualification-r1"
    a = json.loads((logs / "final.json").read_text())
    assert a["status"] == "passed" and sum(g["checks"] for g in a["gates"]) == 36
    dest = WS / "tmp/ck-plugin-b076/export-r1"
    dest.mkdir()
    with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
        assert not verify_hashes(a["protected_sha256"])
        assert all(os.readlink(p) == v for p, v in a["links"].items())
        assert (
            subprocess.check_output(["dmesg"])
            == (logs / "kernel-before.log").read_bytes()
        )
        assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
        guards = []
        for p in Path("/run").glob("ck-plugin-b076-*/state.json"):
            state = json.loads(p.read_text())
            assert (
                state["status"] == "acknowledged" and state["boot_id"] == a["boot_id"]
            )
            guards.append(p.parent)
        assert len(guards) == 6
        cohorts = {}
        for profile in ("daily", "rccl"):
            path = WS / f"logs/ck-plugin-b076-r1-private-{profile}/result.json"
            cohorts[profile] = json.loads(path.read_text())
            assert cohorts[profile]["status"] == "passed"
        report = dict(
            status="qualified-bounded-Search-PerfDb-public-replay-private-only",
            checks=36,
            cohorts=cohorts,
            accepted_guards=len(guards),
            protected_paths=len(a["protected_sha256"]),
            links=len(a["links"]),
            boot_id=a["boot_id"],
            kernel=os.uname().release,
            active_runtime_changed=False,
            host105_contacted=False,
            running_owned_jobs=[],
            ordinary_available_bytes=os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize,
        )
        atomic_json(dest / "verification.json", report)
        with tarfile.open(dest / "evidence.tar.gz", "w:gz") as tar:
            for p in sorted((WS / "logs").glob("*ck-plugin-b076*")):
                tar.add(p, arcname="host-logs/" + p.name)
            for p in guards:
                tar.add(p, arcname="guards/" + p.name)
            for p in sorted((WS / "helpers").glob("*b076*")):
                if p.is_file():
                    tar.add(p, arcname="helpers/" + p.name)
            for pattern in ("inputs-r*", "build-r*"):
                for p in sorted((WS / "tmp/ck-plugin-b076").glob(pattern)):
                    tar.add(p, arcname=p.name)
        atomic_json(
            dest / "archive.json",
            dict(
                sha256=sha(dest / "evidence.tar.gz"),
                bytes=(dest / "evidence.tar.gz").stat().st_size,
            ),
        )
    print("CK_PLUGIN_B076_EVIDENCE_PASS", flush=True)


if __name__ == "__main__":
    main()
