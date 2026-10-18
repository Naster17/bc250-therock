#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Seal isolated N2.8 evidence; verify preserved runtime and N2.7 recovery."""

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
    n27 = json.loads((WS / "logs/solver-final-n27/final.json").read_text())
    protected = prior["protected_sha256"] | prior["artifacts"]
    for name in ("gfx1013_schur_candidate.py", "schur_qr_n27.hip", "schur_qr_n27.so"):
        protected[str(WS / "recovery/solver-n28-before" / name)] = n27["source_sha256"][
            name
        ]
    protected[str(WS / "helpers/schur_n27_reference.py")] = n27["source_sha256"][
        "gfx1013_schur_candidate.py"
    ]
    protected[str(WS / "helpers/schur_qr_n27.so")] = n27["source_sha256"][
        "schur_qr_n27.so"
    ]
    protected["/opt/bc250-stack/current/runtime.json"] = n27["runtime_metadata_sha256"]
    for path, digest in protected.items():
        assert sha(Path(path)) == digest, path
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    labels = ["solver-control-n28-r2"] + [
        f"solver-{group}-n28"
        for group in ("n23", "n24", "scale", "boundary", "failure")
    ]
    gates = []
    for label in labels + ["solver-control-n28"]:
        logs = WS / "logs" / label
        expected_code = 1 if label == "solver-control-n28" else 0
        exit_info = json.loads((logs / "exit.json").read_text())
        assert exit_info == dict(
            returncode=expected_code, kernel_log_unchanged=True
        ), label
        state = json.loads((Path("/run") / label / "state.json").read_text())
        assert state["status"] == "acknowledged" and state["boot_id"] == boot
        assert state["timeout_seconds"] == 600
        if not expected_code:
            result = json.loads((logs / "result.json").read_text())
            assert result["status"] == "passed", label
        gates.append(
            dict(
                label=label,
                returncode=expected_code,
                guard_status=state["status"],
                kernel_log_unchanged=True,
            )
        )
    result = json.loads((WS / "logs/solver-control-n28-r2/result.json").read_text())
    for name, digest in result["hashes"].items():
        assert sha(WS / "helpers" / name) == digest, name
    names = list(result["hashes"]) + [
        "validate_solver_n28.py",
        "validate_solver_n28_regression.py",
        "build_schur_n28.sh",
        "finalize_solver_n28_on_host.py",
    ]
    report = dict(
        status="passed",
        source_sha256={n: sha(WS / "helpers" / n) for n in names},
        gates=gates,
        protected_sha256=protected,
        protected_unchanged=True,
        code_contract="Explicit device_control backend; bounded one QR/launch, five-int D2H status/launch; default reference and production host LAPACK unchanged",
        initial_harness_failure="Direct QR matrix comparison ignored nonunique column phases near singular shifted blocks; phase-aligned T/Q and reconstructed similarity use unchanged bounds",
        kernel=os.uname().release,
        boot_id=boot,
        production_dispatch_unchanged=True,
        running_agent_jobs=[],
        disk_available_bytes=os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize,
    )
    (WS / "logs/solver-control-n28-r2/final.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print("N28_GATES_AND_PROTECTED_HASHES_PASS", len(protected), flush=True)


if __name__ == "__main__":
    main()
