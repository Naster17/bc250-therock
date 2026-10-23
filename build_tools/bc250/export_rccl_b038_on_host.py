#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export owned B03.8 logs, recovery copies and the small packaged RCCL payload."""

import json
import os
from pathlib import Path
import subprocess
import tarfile

from deploy_torch_a04_on_host import atomic_json, sha
from package_rccl_b038_on_host import PROFILE, REGISTRY
from qualify_training_n33_on_host import WS


SOURCES = (
    "bc250_torch.py",
    "update_dev_prefix.py",
    "package_rccl_b038_on_host.py",
    "qualify_rccl_b038_on_host.py",
    "qualify_rccl_b038_context_on_host.py",
    "export_rccl_b038_on_host.py",
    "validate_rccl_b038.py",
    "validate_rccl_b035.py",
    "validate_rccl_b036.py",
    "validate_rccl_b037.py",
)


def main():
    assert os.geteuid() == 0
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    final = json.loads((WS / "logs/rccl-b038-qualification-r1/final.json").read_text())
    context = json.loads((WS / "logs/rccl-b038-context.json").read_text())
    assert final["status"] == context["status"] == "passed"
    for name, digest in final["protected_sha256"].items():
        assert sha(Path(name)) == digest, name
    manifest = json.loads((PROFILE / "runtime.json").read_text())
    destination = WS / "rccl-b038"
    destination.mkdir(exist_ok=True)
    report = dict(
        status="passed",
        gates=len(final["gates"]),
        checks=sum(g["checks"] for g in final["gates"]),
        context_checks=context["gate"]["checks"],
        cpu_refusals=len(context["refusals"]),
        daily_lock_exclusion=context["daily_lock_exclusion"],
        protected_paths=len(final["protected_sha256"]),
        changed_hashes=final["changed_hashes"],
        links_unchanged=final["links_unchanged"],
        kernel_log_unchanged_entire_batch=final["kernel_log_unchanged_entire_batch"],
        boot_id=final["boot_id"],
        boot_unchanged=final["boot_unchanged"],
        daily_runtime_sha256=final["daily_runtime_sha256"],
        profile_sha256=sha(PROFILE / "runtime.json"),
        profile=str(PROFILE),
        registry=str(REGISTRY),
        registry_target=os.readlink(REGISTRY),
        torch_revision=manifest["torch_revision"],
        image=manifest["image"],
        native_identity_sha256=manifest["rccl_single_profile"]["identity_sha256"],
        source_sha256={name: sha(WS / "helpers" / name) for name in SOURCES},
        guards=[
            dict(label=g["label"], status=g["guard"]["status"])
            for g in [*final["gates"], context["gate"]]
        ],
        running_containers=[],
        native_build_required=False,
        host105_touched=False,
        profile_links={
            p.name: os.readlink(p) for p in PROFILE.iterdir() if p.is_symlink()
        },
        disk_available_bytes=os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize,
    )
    assert report["gates"] == 26 and report["checks"] == 1461
    assert report["context_checks"] == 5 and report["cpu_refusals"] == 9
    assert len(report["guards"]) == 27 and all(
        g["status"] == "acknowledged" for g in report["guards"]
    )
    atomic_json(destination / "verification.json", report)
    archive = destination / "evidence.tar.gz"
    assert not archive.exists()
    with tarfile.open(archive, "w:gz") as tar:
        for p in sorted((WS / "logs").glob("rccl-b038-*")):
            tar.add(p, arcname="logs/" + p.name)
        tar.add(WS / "recovery/rccl-b038-before", arcname="recovery")
        for name in SOURCES:
            tar.add(WS / "helpers" / name, arcname="helpers/" + name)
        tar.add(PROFILE / "runtime.json", arcname="profile/runtime.json")
        tar.add(PROFILE / "rccl", arcname="profile/rccl")
        for row in report["guards"]:
            tar.add(Path("/run") / row["label"], arcname="guards/" + row["label"])
        tar.add(destination / "verification.json", arcname="verification.json")
    atomic_json(
        destination / "archive.json",
        dict(sha256=sha(archive), bytes=archive.stat().st_size),
    )
    for p in destination.iterdir():
        os.chown(p, manifest["uid"], manifest["gid"])
    print("RCCL_B038_EXPORT_PASS", archive, flush=True)


if __name__ == "__main__":
    main()
