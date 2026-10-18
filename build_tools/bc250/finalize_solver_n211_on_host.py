#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Seal explicit-session N2.11 evidence; verify unchanged native runtime/recovery."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

WS = Path("/home/nik/bc250work/rocm-10.2-gfx1013")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    assert os.geteuid() == 0
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    prior = json.loads((WS / "logs/solver-q-n210/final.json").read_text())
    protected = dict(prior["protected_sha256"])
    old = prior["source_sha256"]["gfx1013_schur_candidate.py"]
    protected[str(WS / "recovery/solver-n211-before/gfx1013_schur_candidate.py")] = old
    protected[str(WS / "helpers/schur_n210_reference.py")] = old
    protected[str(WS / "helpers/schur_q_n210.so")] = prior["source_sha256"][
        "schur_q_n210.so"
    ]
    for path, digest in protected.items():
        assert sha(Path(path)) == digest, path
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    labels = ["solver-setup-n211-before", "solver-session-n211"] + [
        f"solver-{group}-n211"
        for group in ("n23", "n24", "scale", "boundary", "failure")
    ]
    gates = []
    for label in labels:
        logs = WS / "logs" / label
        assert json.loads((logs / "exit.json").read_text()) == dict(
            returncode=0, kernel_log_unchanged=True
        ), label
        assert (
            json.loads((logs / "result.json").read_text())["status"] == "passed"
        ), label
        state = json.loads((Path("/run") / label / "state.json").read_text())
        assert state["status"] == "acknowledged" and state["boot_id"] == boot
        assert state["timeout_seconds"] == 600
        gates.append(
            dict(label=label, guard_status=state["status"], kernel_log_unchanged=True)
        )
    result = json.loads((WS / "logs/solver-session-n211/result.json").read_text())
    for name, digest in result["hashes"].items():
        assert sha(WS / "helpers" / name) == digest, name
    names = list(result["hashes"]) + [
        "validate_solver_n211.py",
        "validate_solver_n211_regression.py",
        "finalize_solver_n211_on_host.py",
    ]
    report = dict(
        status="passed",
        source_sha256={n: sha(WS / "helpers" / n) for n in names},
        gates=gates,
        protected_sha256=protected,
        protected_unchanged=True,
        code_contract="Explicit caller-owned GEHRD session, one device/stream/host-thread/PID, owned inputs, close syncs pinned stream before destruction; no global handle cache/default change",
        kernel=os.uname().release,
        boot_id=boot,
        production_dispatch_unchanged=True,
        running_agent_jobs=[],
        disk_available_bytes=os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize,
    )
    (WS / "logs/solver-session-n211/final.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print("N211_GATES_AND_PROTECTED_HASHES_PASS", len(protected), flush=True)


if __name__ == "__main__":
    main()
