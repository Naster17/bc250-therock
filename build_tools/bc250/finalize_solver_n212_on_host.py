#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Seal bounded N2.12 recovery evidence and unchanged protected runtime."""

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
    prior = json.loads((WS / "logs/solver-session-n211/final.json").read_text())
    maintenance = json.loads(
        (WS / "logs/storage-refresh-20261018/final.json").read_text()
    )
    protected = dict(maintenance["protected_sha256"])
    old = prior["source_sha256"]["gfx1013_schur_candidate.py"]
    protected[str(WS / "recovery/solver-n212-before/gfx1013_schur_candidate.py")] = old
    protected[str(WS / "helpers/schur_n211_reference.py")] = old
    protected[str(WS / "helpers/gfx1013_gehrd_session.py")] = prior["source_sha256"][
        "gfx1013_gehrd_session.py"
    ]
    for path, digest in protected.items():
        assert sha(Path(path)) == digest, path
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    labels = [
        "solver-recovery-n212-before",
        "solver-backsub-n212-r2",
        "solver-native-n212",
    ] + [
        f"solver-{group}-n212"
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
    result = json.loads((WS / "logs/solver-backsub-n212-r2/result.json").read_text())
    for name, digest in result["hashes"].items():
        assert sha(WS / "helpers" / name) == digest, name
    names = list(result["hashes"]) + [
        "validate_solver_n212.py",
        "validate_solver_n212_native.py",
        "validate_solver_n212_regression.py",
        "build_schur_n212.sh",
        "finalize_solver_n212_on_host.py",
    ]
    failed_logs = WS / "logs/solver-backsub-n212"
    assert json.loads((failed_logs / "exit.json").read_text()) == dict(
        returncode=1, kernel_log_unchanged=True
    )
    failed_guard = json.loads((Path("/run/solver-backsub-n212/state.json")).read_text())
    assert failed_guard["status"] == "acknowledged" and failed_guard["boot_id"] == boot
    assert failed_guard["timeout_seconds"] == 600
    initial = (WS / "logs/solver-recovery-n212-before/kernel-before.log").read_bytes()
    assert subprocess.check_output(["dmesg"]) == initial
    report = dict(
        status="passed",
        source_sha256={n: sha(WS / "helpers" / n) for n in names},
        gates=gates,
        protected_sha256=protected,
        protected_unchanged=True,
        code_contract="Explicit recovery_backend=device fuses only normalized triangular solve, cap16 complex64/128; common fixed finite/unitary/separation/rescaling/unitnorm/residual gates and reference/host production defaults retained",
        failed_capture=dict(
            label="solver-backsub-n212",
            numerical_checks_completed=True,
            missing_gpu_events_after_cpu_only_first=True,
            guard_status="acknowledged",
            kernel_log_unchanged=True,
        ),
        profiler_issue_unresolved=True,
        kernel_log_unchanged_entire_batch=True,
        kernel=os.uname().release,
        boot_id=boot,
        production_dispatch_unchanged=True,
        running_agent_jobs=[],
        disk_available_bytes=os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize,
    )
    (WS / "logs/solver-backsub-n212-r2/final.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print("N212_GATES_AND_PROTECTED_HASHES_PASS", len(protected), flush=True)


if __name__ == "__main__":
    main()
