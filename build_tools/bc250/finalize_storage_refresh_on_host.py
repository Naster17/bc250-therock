#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify cleanup journal, unchanged protected runtime and bounded GPU gates."""

import json
import os
from pathlib import Path
import subprocess

from cleanup_storage_refresh import (
    LOGS,
    WS,
    check_gate,
    check_protected,
    container_rows,
    free,
    images,
    save,
    sha,
)


def main() -> None:
    assert os.geteuid() == 0
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    plan = json.loads((LOGS / "plan.json").read_text())
    applied = json.loads((LOGS / "applied.json").read_text())
    assert applied["status"] == "passed"
    check_protected(plan["protected_sha256"])
    assert all(not Path(r["path"]).exists() for r in plan["files"])
    rows = container_rows()
    assert all(r in rows for r in plan["protected_containers"])
    assert not any(r["id"] in {v["id"] for v in plan["containers"]} for r in rows)
    assert images() == plan["images_before"]
    for p, target in plan["links"].items():
        assert str(Path(p).readlink()) == target
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    labels = [
        "storage-refresh-current-before",
        "storage-refresh-current-after",
        "storage-refresh-fallback-after",
        "storage-refresh-current-fresh-jit",
        "storage-refresh-fallback-fresh-jit",
    ]
    for label in labels:
        check_gate(label)
        state = json.loads((Path("/run") / label / "state.json").read_text())
        assert state["boot_id"] == boot
    before = (WS / "logs/storage-refresh-current-before/kernel-before.log").read_bytes()
    assert (
        subprocess.check_output(["dmesg"]) == before
    ), "Entire cleanup kernel log must remain unchanged"
    counts = {}
    for r in plan["files"]:
        bucket = counts.setdefault(r["kind"], dict(files=0, allocated_bytes=0))
        bucket["files"] += 1
        bucket["allocated_bytes"] += r["stamp"]["blocks"] * 512
    result = dict(
        status="passed",
        recovered_bytes=applied["recovered_bytes"],
        free_before_bytes=applied["free_before_bytes"],
        free_after_bytes=free(),
        files_by_kind=counts,
        removed_containers=len(plan["containers"]),
        protected_sha256=plan["protected_sha256"],
        images_unchanged=True,
        protected_containers_unchanged=True,
        links_unchanged=True,
        kernel_log_unchanged_entire_batch=True,
        gates=labels,
        checks_per_gate=8,
        guard_timeout_seconds=600,
        boot_id=boot,
        kernel=os.uname().release,
        running_agent_jobs=[],
        next="N2.12 unchanged: warm-session recovery profile and bounded native triangular back-substitution",
        source_sha256={
            n: sha(WS / "helpers" / n)
            for n in [
                "cleanup_storage_refresh.py",
                "validate_storage_refresh_runtime.py",
                "finalize_storage_refresh_on_host.py",
            ]
        },
    )
    save("final.json", result)
    print(
        "STORAGE_REFRESH_FINAL_PASS",
        result["recovered_bytes"],
        result["free_after_bytes"],
        len(plan["protected_sha256"]),
        flush=True,
    )


if __name__ == "__main__":
    main()
