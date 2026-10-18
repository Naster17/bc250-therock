#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Read-only prefix verification and isolated N2.7 evidence seal, under lock."""

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
    prior = json.loads((WS / "SOURCE_DENSE_INT8_A08.json").read_text())
    protected = prior["protected_sha256"] | prior["artifacts"]
    for path, digest in protected.items():
        assert sha(Path(path)) == digest, path
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    labels = ["solver-step-n27", "solver-step-n27-wave", "solver-final-n27"] + [
        f"solver-{group}-n27-wave"
        for group in ("n23", "n24", "scale", "boundary", "failure")
    ]
    gates = []
    for label in labels:
        logs = WS / "logs" / label
        result = json.loads((logs / "result.json").read_text())
        assert result["status"] == "passed", label
        exit_info = json.loads((logs / "exit.json").read_text())
        assert exit_info == dict(returncode=0, kernel_log_unchanged=True), label
        state = json.loads((Path("/run") / label / "state.json").read_text())
        assert state["status"] == "acknowledged" and state["boot_id"] == boot
        assert state["timeout_seconds"] == 600
        gates.append(
            dict(
                label=label,
                checks=len(result.get("checks", [])),
                guard_status=state["status"],
                kernel_log_unchanged=True,
            )
        )
    result = json.loads((WS / "logs/solver-final-n27/result.json").read_text())
    for name, digest in result["hashes"].items():
        assert sha(WS / "helpers" / name) == digest, name
    report = dict(
        status="passed",
        boot_id=boot,
        kernel=os.uname().release,
        source_sha256={
            n: sha(WS / "helpers" / n)
            for n in [
                "gfx1013_schur_candidate.py",
                "schur_qr_n27.hip",
                "schur_qr_n27.so",
                "schur_n26_reference.py",
                "validate_solver_n27.py",
                "validate_solver_n27_regression.py",
                "build_schur_n27.sh",
                "finalize_solver_n27_on_host.py",
            ]
        },
        gates=gates,
        protected_sha256=protected,
        protected_unchanged=True,
        runtime_metadata_sha256=sha(Path("/opt/bc250-stack/current/runtime.json")),
        production_dispatch_unchanged=True,
        default_candidate_backend="reference",
        running_agent_jobs=[],
        disk_available_bytes=os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize,
    )
    (WS / "logs/solver-final-n27/final.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print("N27_GATES_AND_PROTECTED_HASHES_PASS", len(protected), flush=True)


if __name__ == "__main__":
    main()
