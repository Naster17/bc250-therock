#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Seal final target identities after the post-retirement normal r3 boot."""

import json
import os
import shutil
import subprocess
from pathlib import Path

from migrate_reference_host import digest
from ready_migrated_kernel import note_id

WS = Path("/home/nik/bc250work/migration-105-20261023")
RELEASE = Path("/opt/bc250-stack/releases/10.2-gfx1013-20261023-host105")


def main() -> None:
    assert os.geteuid() == 0 and os.uname().nodename == "bc500"
    assert os.uname().release == "7.2.10-bc250rocm-r3"
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    readiness = json.loads(
        (
            Path("/var/lib/bc250-stack/host105")
            / ("promoted-boot-" + boot)
            / "result.json"
        ).read_text()
    )
    assert (
        readiness["status"] == "passed"
        and readiness["saved_fallback_before_readiness"] == "bc250-rocm-r2-kernel"
    )
    assert (
        json.loads((WS / "finish.json").read_text())["status"]
        == "passed_final_boot_pending"
    )
    assert (
        json.loads((WS / "recovery-validation.json").read_text())["status"] == "passed"
    )
    assert (
        json.loads((WS / "final-ordinary-torch.json").read_text())["status"] == "passed"
    )
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    variables = dict(
        line.split("=", 1)
        for line in subprocess.check_output(
            ["grub-editenv", "/boot/grub/grubenv", "list"], text=True
        ).splitlines()
        if "=" in line
    )
    assert not variables.get("next_entry")
    assert (
        "GRUB_DEFAULT=bc250-host105-r3-release" in Path("/etc/default/grub").read_text()
    )
    fallback = json.loads((WS / "before.json").read_text())["fallback_hashes"]
    assert all(digest(Path(name)) == sha for name, sha in fallback.items())
    assert (
        digest(
            Path(
                "/lib/modules/7.1.5-bc250rocm-r2/kernel/drivers/gpu/drm/amd/amdgpu/amdgpu.ko.gz"
            )
        )
        == "88e9e465a0fb668075e10c83ccabf473d2482095cf7f27c7ee06b77f24cd2799"
    )
    manifest = json.loads((RELEASE / "runtime.json").read_text())
    for name, sha in manifest["operational_boot_tools"]["sha256"].items():
        assert digest(RELEASE / "healthchecks" / name) == sha
    guards = []
    for base in (WS, Path("/var/lib/bc250-stack/host105")):
        for path in base.rglob("state.json"):
            state = json.loads(path.read_text())
            if "deadline_monotonic" not in state:
                continue
            assert state["status"] == "acknowledged" or (
                state["status"] == "expired"
                and state["phase"] == "migration-fallback-test"
            ), str(path)
            guards.append(
                {"path": str(path), "phase": state["phase"], "status": state["status"]}
            )
    # These are unmodified public upstream download archives, not unique work.
    cache = Path("/home/nik/bc250work/alpine-rocm")
    retired_cache = {}
    if cache.exists():
        sources = cache / "sources"
        assert sorted(p.name for p in sources.iterdir()) == [
            "linux-6.18.tar.xz",
            "patch-6.18.50.patch.xz",
        ]
        retired_cache = {str(p): digest(p) for p in sources.iterdir()}
        shutil.rmtree(cache)
    proof = {
        "status": "complete",
        "hostname": os.uname().nodename,
        "kernel": os.uname().release,
        "boot_id": boot,
        "release": str(RELEASE),
        "image": manifest["image"],
        "runtime_sha256": digest(RELEASE / "runtime.json"),
        "kernel_build_id": note_id(Path("/sys/kernel/notes").read_bytes()),
        "module_build_id": note_id(
            Path("/sys/module/amdgpu/notes/.note.gnu.build-id").read_bytes()
        ),
        "torch_revision": manifest["torch_revision"],
        "hip_revision": manifest["rocm_systems_revision"],
        "readiness": readiness,
        "main_qualification": json.loads(
            (WS / "qualification-r3/final.json").read_text()
        ),
        "retirement": json.loads((WS / "retirement.json").read_text()),
        "fallback_hashes": fallback,
        "retired_public_downloads": retired_cache,
        "guard_states": guards,
        "disk_free_bytes": shutil.disk_usage("/").free,
        "limits": [
            "Bounded one-board tests; no universal correctness, performance, long-soak or multi-board qualification",
            "Historical Python CPU fault remains unresolved, distinct from VM-heap/CSR/hostcall fixes",
            "Idle deadline emergency return to r2 tested; arbitrary/pre-watchdog hardware hangs unqualified",
        ],
    }
    (WS / "final.json").write_text(json.dumps(proof, indent=2) + "\n")
    (WS / "migration-status.json").write_text(
        json.dumps(
            {
                "phase": "hardware migration complete",
                "final_report": str(WS / "final.json"),
                "next": "Commit tools/evidence and canonical flow; resume B03.8",
            },
            indent=2,
        )
        + "\n"
    )
    os.sync()
    print("HOST105_MIGRATION_COMPLETE", boot, proof["disk_free_bytes"], flush=True)


if __name__ == "__main__":
    main()
