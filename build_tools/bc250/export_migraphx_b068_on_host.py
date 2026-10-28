#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Seal the installed opt-in SDK, guarded qualification and reversible registration evidence."""
import json
import os
from pathlib import Path
import subprocess
import tarfile

from bc250_migraphx import REGISTRY, NAME, registered
from bc250_torch import LaunchOptions, runtime_lock
from build_migraphx_b067_on_host import verify_hashes
from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS


def main():
    logs = WS / "logs/migraphx-b068-qualification-r1"
    q = json.loads((logs / "final.json").read_text())
    assert q["status"] == "passed" and sum(g["checks"] for g in q["gates"]) == 150
    assert len(q["cpu_checks"]) == 14
    dest = WS / "tmp/migraphx-b068/export-r1"
    dest.mkdir()
    with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
        record, sdk = registered(LaunchOptions())
        registered(LaunchOptions(runtime_profile="rccl-single"))
        assert not verify_hashes(q["protected_sha256"])
        assert all(os.readlink(p) == v for p, v in q["links"].items())
        assert (
            subprocess.check_output(["dmesg"])
            == (logs / "kernel-before.log").read_bytes()
        )
        assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
        user = json.loads((WS / "logs/migraphx-b068-user-info.json").read_text())
        assert user == record
        guards = {}
        for entry in q["gates"]:
            path = Path("/run") / entry["label"]
            state = json.loads((path / "state.json").read_text())
            assert (
                state["status"] == "acknowledged" and state["boot_id"] == q["boot_id"]
            )
            guards[entry["label"]] = path
        index = json.loads((sdk / "share/bc250-sdk/sha256.json").read_text())
        report = dict(
            status="qualified-installed-opt-in-sdk",
            checks=165,
            ordinary_guarded_checks=150,
            cpu_registration_checks=14,
            ordinary_user_elevation_checks=1,
            gates=len(q["gates"]),
            acknowledged_guards=len(guards),
            gpu_graph_evaluations=68,
            sdk=str(sdk),
            registration=record,
            installed_sdk_files=len(index["files"]) + 1,
            installed_sdk_links=len(index["symlinks"]),
            native_preserved_paths=len(q["protected_sha256"]),
            preserved_links=len(q["links"]),
            aot_caches=q["aot_caches"],
            boot_id=q["boot_id"],
            kernel=os.uname().release,
            active_runtime_changed=False,
            host105_contacted=False,
            kernel_log_unchanged=True,
            running_agent_jobs=[],
            ordinary_free_bytes=os.statvfs("/opt").f_bavail
            * os.statvfs("/opt").f_frsize,
        )
        atomic_json(dest / "verification.json", report)
        with tarfile.open(dest / "evidence.tar.gz", "w:gz") as tar:
            for path in sorted((WS / "logs").glob("*migraphx-b068*")):
                tar.add(path, arcname="host-logs/" + path.name)
            for name, path in guards.items():
                tar.add(path, arcname="guards/" + name)
            for path in sorted((WS / "helpers").glob("*migraphx*b068*")):
                if path.is_file():
                    tar.add(path, arcname="helpers/" + path.name)
            tar.add(
                WS / "helpers/bc250_migraphx.py", arcname="helpers/bc250_migraphx.py"
            )
            tar.add(REGISTRY, arcname="registration/migraphx.json")
            tar.add(
                REGISTRY.parent / (NAME + ".json"),
                arcname="registration/versioned.json",
            )
            for path in sorted(
                (Path("/opt/bc250-stack/tools/migraphx-b068")).glob("*.py")
            ):
                tar.add(path, arcname="installed-tools/" + path.name)
            for path in sorted(
                Path(
                    "/opt/bc250-stack/sdk-fixtures/b068/application with spaces"
                ).iterdir()
            ):
                tar.add(path, arcname="external-fixture/" + path.name)
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
