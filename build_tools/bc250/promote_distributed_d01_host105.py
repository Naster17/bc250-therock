#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Atomically promote the target-qualified daily update under .105's control EX lock."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollback", action="store_true")
    a = p.parse_args()
    assert os.geteuid() == 0 and os.uname().nodename == "bc500"
    current = Path("/opt/bc250-stack/current")
    dest = Path("/opt/bc250-stack/releases/10.2-gfx1013-20261104-d01")
    old = Path("/opt/bc250-stack/releases/10.2-gfx1013-20261023-host105")
    recovery = Path("/home/nik/bc250work/distributed-d01-recovery")
    if a.rollback:
        r = json.loads((recovery / "activation.json").read_text())
        assert (
            current.resolve() == dest
            and sha(old / "runtime.json") == r["old_runtime_sha256"]
        )
        temp = current.with_name("current-d01-rollback")
        temp.symlink_to(old)
        temp.replace(current)
        return
    assert current.resolve() == old
    report = Path(
        "/home/nik/bc250work/distributed-d01-target-qualification-r1/final.json"
    )
    r = json.loads(report.read_text())
    assert r["status"] == "passed" and r["checks"] == 656 and len(r["gates"]) == 9
    assert (
        r["boot_id"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        and sha(dest / "runtime.json") == r["runtime_sha256"]
    )
    recovery.mkdir()
    (recovery / "qualified-runtime.json").write_bytes(
        (dest / "runtime.json").read_bytes()
    )
    data = json.loads((dest / "runtime.json").read_text())
    data["target_qualification"] = dict(
        status="passed",
        checks=656,
        report=str(report),
        report_sha256=sha(report),
        qualified_payload_runtime_sha256=r["runtime_sha256"],
    )
    data["operational_boot_tools"] = json.loads((old / "runtime.json").read_text()).get(
        "operational_boot_tools", {}
    )
    (dest / "runtime.json").write_text(json.dumps(data, indent=2) + "\n")
    profile = dest / "rccl-profile/runtime.json"
    (recovery / "qualified-rccl-runtime.json").write_bytes(profile.read_bytes())
    rccl = json.loads(profile.read_text())
    rccl["target_qualification"] = (
        "D02/D03/D04 bounded two-host RCCL/DDP/FSDP/checkpoint references passed; broader transport/scaling unqualified"
    )
    profile.write_text(json.dumps(rccl, indent=2) + "\n")
    registry = Path("/opt/bc250-stack/distributed/d01.json")
    (recovery / "distributed-registry-before.json").write_bytes(registry.read_bytes())
    registered = json.loads(registry.read_text())
    registered["runtime_sha256"] = sha(profile)
    registered["old_daily_unchanged"] = False
    registered[
        "scope"
    ] += "; bounded nested FSDP/AdamW shards and same-world checkpoint resume passed"
    replacement = registry.with_suffix(".new")
    replacement.write_text(json.dumps(registered, indent=2) + "\n")
    replacement.chmod(0o444)
    replacement.replace(registry)
    record = dict(
        old_current=str(old),
        new_current=str(dest),
        old_runtime_sha256=sha(old / "runtime.json"),
        new_runtime_sha256=sha(dest / "runtime.json"),
        report_sha256=sha(report),
        old_boot_health_release_preserved=True,
        kernel_unchanged=os.uname().release,
    )
    (recovery / "activation.json").write_text(json.dumps(record, indent=2) + "\n")
    temp = current.with_name("current-d01-new")
    temp.symlink_to(dest)
    temp.replace(current)


if __name__ == "__main__":
    main()
