#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Seal relocation, identity and extracted-artifact qualification results and their unchanged native-runtime baseline."""
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
    parser.add_argument("--attempt", choices=("r1", "r2", "r3"), required=True)
    parser.add_argument("--package-attempt", choices=("r1", "r2", "r3"), required=True)
    args = parser.parse_args()
    logs = WS / ("logs/ck-plugin-b0712-qualification-" + args.attempt)
    a = json.loads((logs / "final.json").read_text())
    assert a["status"] == "passed" and sum(g["checks"] for g in a["gates"]) == 78
    dest = WS / ("tmp/ck-plugin-b0712/export-" + args.attempt)
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
        for p in Path("/run").glob("ck-plugin-b0712-*/state.json"):
            state = json.loads(p.read_text())
            assert (
                state["status"] == "acknowledged" and state["boot_id"] == a["boot_id"]
            )
            guards.append(p.parent)
        assert (
            len(
                [
                    p
                    for p in guards
                    if p.name.startswith("ck-plugin-b0712-" + args.attempt + "-")
                ]
            )
            == 10
        )
        cohorts = {}
        for number in (0, 1):
            for profile in ("daily", "rccl-single"):
                path = (
                    WS
                    / f"logs/ck-plugin-b0712-{args.attempt}-extract{number}-{profile}/result.json"
                )
                value = json.loads(path.read_text())
                assert value["status"] == "passed"
                cohorts[str(number) + "-" + profile] = value
        report = dict(
            status="qualified-opt-in-artifact-not-installed",
            checks=78,
            cohorts=cohorts,
            accepted_guards=10,
            total_acknowledged_guards=len(guards),
            protected_paths=len(a["protected_sha256"]),
            links=len(a["links"]),
            boot_id=a["boot_id"],
            kernel=os.uname().release,
            active_runtime_changed=False,
            host105_contacted=False,
            running_owned_jobs=[],
            ordinary_available_bytes=os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize,
        )
        package = json.loads(
            (
                WS
                / (
                    "tmp/ck-plugin-b0712/package-"
                    + args.package_attempt
                    + "/final.json"
                )
            ).read_text()
        )
        report["package"] = {
            k: package[k]
            for k in (
                "archive",
                "archive_sha256",
                "archive_bytes",
                "roots",
                "payload_files",
                "payload_bytes",
            )
        }
        atomic_json(dest / "verification.json", report)
        with tarfile.open(dest / "evidence.tar.gz", "w:gz") as tar:
            for p in sorted((WS / "logs").glob("*ck-plugin-b0712*")):

                def retain(info):
                    return (
                        None
                        if any(
                            part.startswith("bad-") for part in Path(info.name).parts
                        )
                        else info
                    )

                tar.add(p, arcname="host-logs/" + p.name, filter=retain)
                if p.is_dir():
                    for manifest in p.glob("bad-*/manifest.json"):
                        tar.add(
                            manifest,
                            arcname="host-logs/"
                            + p.name
                            + "/"
                            + str(manifest.relative_to(p)),
                        )
            for p in guards:
                tar.add(p, arcname="guards/" + p.name)
            for p in sorted((WS / "helpers").glob("*b0712*")):
                if p.is_file():
                    tar.add(p, arcname="helpers/" + p.name)
            for folder in (WS / "tmp/ck-plugin-b0712").glob("package-r*"):
                for p in folder.glob("*.json"):
                    tar.add(p, arcname=folder.name + "/" + p.name)
            for p in (WS / "tmp/ck-plugin-b0712/resources").iterdir():
                if p.is_file():
                    tar.add(p, arcname="resources/" + p.name)
        atomic_json(
            dest / "archive.json",
            dict(
                sha256=sha(dest / "evidence.tar.gz"),
                bytes=(dest / "evidence.tar.gz").stat().st_size,
            ),
        )
    print("CK_PLUGIN_B0712_EVIDENCE_PASS", flush=True)


if __name__ == "__main__":
    main()
