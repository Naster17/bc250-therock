#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Seal isolated N2.10 Q-formation evidence and unchanged runtime/recovery."""

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
    prior = json.loads((WS / "logs/solver-chunk-n29/final.json").read_text())
    protected = dict(prior["protected_sha256"])
    old = prior["source_sha256"]["gfx1013_schur_candidate.py"]
    protected[str(WS / "recovery/solver-n210-before/gfx1013_schur_candidate.py")] = old
    protected[str(WS / "helpers/schur_n29_reference.py")] = old
    protected[str(WS / "helpers/schur_chunk_n29.so")] = prior["source_sha256"][
        "schur_chunk_n29.so"
    ]
    for path, digest in protected.items():
        assert sha(Path(path)) == digest, path
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    labels = ["solver-prep-n210-before", "solver-q-n210"] + [
        f"solver-{group}-n210"
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
    result = json.loads((WS / "logs/solver-q-n210/result.json").read_text())
    for name, digest in result["hashes"].items():
        assert sha(WS / "helpers" / name) == digest, name
    names = list(result["hashes"]) + [
        "validate_solver_n210.py",
        "validate_solver_n210_regression.py",
        "build_schur_n210.sh",
        "finalize_solver_n210_on_host.py",
    ]
    report = dict(
        status="passed",
        source_sha256={n: sha(WS / "helpers" / n) for n in names},
        gates=gates,
        protected_sha256=protected,
        protected_unchanged=True,
        code_contract="Explicit q_backend=device for cap16 complex GEHRD reflector-Q formation; reference Q and production host eig defaults unchanged",
        kernel=os.uname().release,
        boot_id=boot,
        production_dispatch_unchanged=True,
        running_agent_jobs=[],
        disk_available_bytes=os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize,
    )
    (WS / "logs/solver-q-n210/final.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print("N210_GATES_AND_PROTECTED_HASHES_PASS", len(protected), flush=True)


if __name__ == "__main__":
    main()
