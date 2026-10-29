#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Seal bounded FP16 public tuning/replay and paired timing with the preserved runtime baseline."""
import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attempt", required=True, choices=("r1", "r2", "r3", "r4", "r5")
    )
    args = parser.parse_args()
    logs = WS / ("logs/ck-plugin-b0711-qualification-" + args.attempt)
    a = json.loads((logs / "final.json").read_text())
    assert a["status"] == "passed" and sum(g["checks"] for g in a["gates"]) == 80
    dest = WS / ("tmp/ck-plugin-b0711/export-" + args.attempt)
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
        for p in Path("/run").glob("ck-plugin-b0711-" + args.attempt + "-*/state.json"):
            state = json.loads(p.read_text())
            assert (
                state["status"] == "acknowledged" and state["boot_id"] == a["boot_id"]
            )
            guards.append(p.parent)
        assert len(guards) == 10
        cohorts = {}
        for name in (
            "private-daily",
            "private-rccl",
            "private-contracts-daily",
            "private-contracts-rccl",
            "installed-daily",
            "installed-rccl",
        ):
            value = json.loads(
                (
                    WS / f"logs/ck-plugin-b0711-{args.attempt}-{name}/result.json"
                ).read_text()
            )
            assert value["status"] == "passed"
            cohorts[name] = value
        report = dict(
            status="qualified-isolated-FP16-public-timing-native-tuning-not-installed",
            checks=80,
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
            for p in sorted((WS / "logs").glob("*ck-plugin-b0711*")):
                tar.add(p, arcname="host-logs/" + p.name)
            for p in guards:
                tar.add(p, arcname="guards/" + p.name)
            for p in sorted((WS / "helpers").glob("*b0711*")):
                if p.is_file():
                    tar.add(p, arcname="helpers/" + p.name)
            for p in (WS / "tmp/ck-plugin-b0711").glob("*audit*.json"):
                tar.add(p, arcname="audits/" + p.name)
            native = WS / "tmp/ck-plugin-b0711/native-fix-r2"
            for p in native.iterdir():
                if p.is_file():
                    tar.add(p, arcname="native-fix-r2/" + p.name)
            tar.add(
                native / "miopen-build/lib/libMIOpen.so.1.0",
                arcname="native-fix-r2/libMIOpen.so.1.0",
            )
            tar.add(
                WS / "tmp/ck-plugin-b0711/native-inputs-r1", arcname="native-inputs-r1"
            )
            for p in Path("/run").glob("ck-plugin-b0711-*/state.json"):
                state = json.loads(p.read_text())
                assert state["status"] == "acknowledged"
                if p.parent not in guards:
                    tar.add(p.parent, arcname="earlier-guards/" + p.parent.name)
            for pattern in ("inputs-r*", "build-r*"):
                for p in sorted((WS / "tmp/ck-plugin-b0711").glob(pattern)):
                    tar.add(p, arcname=p.name)
        atomic_json(
            dest / "archive.json",
            dict(
                sha256=sha(dest / "evidence.tar.gz"),
                bytes=(dest / "evidence.tar.gz").stat().st_size,
            ),
        )
    print("CK_PLUGIN_B0711_EVIDENCE_PASS", flush=True)


if __name__ == "__main__":
    main()
