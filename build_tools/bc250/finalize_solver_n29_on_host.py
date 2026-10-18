#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Seal isolated N2.9 chunk evidence and verify unchanged runtime/recovery."""

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
    prior = json.loads((WS / "logs/solver-control-n28-r2/final.json").read_text())
    protected = dict(prior["protected_sha256"])
    for name in (
        "gfx1013_schur_candidate.py",
        "schur_control_n28.hip",
        "schur_control_n28.so",
    ):
        protected[str(WS / "recovery/solver-n29-before" / name)] = prior[
            "source_sha256"
        ][name]
    protected[str(WS / "helpers/schur_n28_reference.py")] = prior["source_sha256"][
        "gfx1013_schur_candidate.py"
    ]
    protected[str(WS / "helpers/schur_control_n28.so")] = prior["source_sha256"][
        "schur_control_n28.so"
    ]
    protected[str(WS / "recovery/solver-n29-unpadded/schur_qr_n27.hip")] = prior[
        "source_sha256"
    ]["schur_qr_n27.hip"]
    for path, digest in protected.items():
        assert sha(Path(path)) == digest, path
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    labels = ["solver-chunk-n29", "solver-chunk-n29-padded"] + [
        f"solver-{group}-n29{suffix}"
        for suffix in ("", "-padded")
        for group in ("n23", "n24", "scale", "boundary", "failure")
    ]
    gates = []
    for label in labels:
        logs = WS / "logs" / label
        assert json.loads((logs / "exit.json").read_text()) == dict(
            returncode=0, kernel_log_unchanged=True
        ), label
        result = json.loads((logs / "result.json").read_text())
        assert result["status"] == "passed", label
        state = json.loads((Path("/run") / label / "state.json").read_text())
        assert state["status"] == "acknowledged" and state["boot_id"] == boot
        assert state["timeout_seconds"] == 600
        gates.append(
            dict(label=label, guard_status=state["status"], kernel_log_unchanged=True)
        )
    result = json.loads((WS / "logs/solver-chunk-n29/result.json").read_text())
    for name, digest in result["hashes"].items():
        assert sha(WS / "helpers" / name) == digest, name
    names = list(result["hashes"]) + [
        "validate_solver_n29.py",
        "validate_solver_n29_regression.py",
        "build_schur_n29.sh",
        "finalize_solver_n29_on_host.py",
    ]
    report = dict(
        status="passed",
        source_sha256={n: sha(WS / "helpers" / n) for n in names},
        gates=gates,
        protected_sha256=protected,
        protected_unchanged=True,
        code_contract="Explicit device_chunked backend; chunk_size1..32/default16, exact total QR budget, immediate terminal/failure stop; reference and production host LAPACK unchanged",
        kernel=os.uname().release,
        boot_id=boot,
        running_agent_jobs=[],
        production_dispatch_unchanged=True,
        disk_available_bytes=os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize,
    )
    (WS / "logs/solver-chunk-n29/final.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print("N29_GATES_AND_PROTECTED_HASHES_PASS", len(protected), flush=True)


if __name__ == "__main__":
    main()
