#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Seal installed sparse triangular references, preserved cohorts and GPU guards."""

import json
import os
from pathlib import Path
import subprocess

from deploy_torch_a04_on_host import atomic_json, check_gate, sha
from qualify_sparse_n213_on_host import CANDIDATE, LOGS, OLD, RELEASE, WS


def main() -> None:
    assert (
        os.geteuid() == 0
        and not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    )
    installed = json.loads((LOGS / "installed.json").read_text())
    candidate = json.loads((LOGS / "candidate.json").read_text())
    assert installed["status"] == candidate["status"] == "passed"
    assert (RELEASE / "torch").resolve() == CANDIDATE
    assert (
        json.loads((RELEASE / "runtime.json").read_text())["torch_revision"]
        == installed["source"]["torch_revision"]
    )
    assert len(installed["protected_after"]) == 121
    for p, d in installed["protected_after"].items():
        assert sha(Path(p)) == d, p
    for p, d in installed["candidate_sha256"].items():
        assert sha(CANDIDATE / p) == d, p
    assert sha(CANDIDATE / "test/test_bc250_sparse_triangular.py") == sha(
        WS / "helpers/test_bc250_sparse_triangular.py"
    )
    for label in candidate["passed_gates"] + installed["installed_gates"]:
        check_gate(label)
    assert len(candidate["passed_gates"]) == len(installed["installed_gates"]) == 9
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    failed = {"sparse-n213-before-edge", "sparse-n213-before-edge-r2"}
    gates = []
    for folder in sorted((WS / "logs").glob("sparse-n213-*")):
        if not (folder / "guard.log").exists():
            continue
        status = json.loads((folder / "exit.json").read_text())
        assert status == dict(
            returncode=int(folder.name in failed), kernel_log_unchanged=True
        ), folder
        state = json.loads((Path("/run") / folder.name / "state.json").read_text())
        assert (
            state["status"] == "acknowledged"
            and state["timeout_seconds"] == 600
            and state["boot_id"] == boot
        )
        gates.append(dict(label=folder.name, **status, guard_status="acknowledged"))
    assert len(gates) == 22
    assert (
        subprocess.check_output(["dmesg"])
        == (WS / "logs/sparse-n213-before-matrix/kernel-before.log").read_bytes()
    )
    before = json.loads(
        (WS / "logs/sparse-n213-before-edge-r2/result.json").read_text()
    )
    assert (
        before["status"] == "failed"
        and len(before["checks"]) == 80
        and len(before["failed"]) > 0
    )
    changed = [
        p.name
        for p in (CANDIDATE / "torch/lib").glob("*.so")
        if sha(p) != sha(OLD / "torch/lib" / p.name)
    ]
    assert "libtorch_hip.so" in changed
    names = [
        "build_sparse_n213_on_host.py",
        "validate_sparse_n213.py",
        "validate_sparse_n213_native.py",
        "test_bc250_sparse_triangular.py",
        "qualify_sparse_n213_on_host.py",
        "finalize_sparse_n213_on_host.py",
    ]
    generated = {
        str(p.relative_to(CANDIDATE)): sha(p)
        for p in (CANDIDATE / "aten/src/ATen/native/sparse/hip").glob("Sparse*")
        if p.name in ("SparseBlasImpl.h", "SparseBlas.cpp", "SparseCsrTensorMath.hip")
    }
    assert len(generated) == 3
    report = dict(
        status="passed",
        source=installed["source"],
        source_sha256={n: sha(WS / "helpers" / n) for n in names},
        candidate_sha256=installed["candidate_sha256"],
        generated_source_sha256=generated,
        protected_sha256=installed["protected_after"],
        gates=gates,
        changed_native_libraries=changed,
        before_failed_pivot_cases=len(before["failed"]),
        new_torch_reference_groups=433,
        native_generic_cases=96,
        native_generic_solve_executions=192,
        kernel_log_unchanged_entire_batch=True,
        kernel=os.uname().release,
        boot_id=boot,
        runtime_sha256=sha(RELEASE / "runtime.json"),
        torch_target=str((RELEASE / "torch").resolve()),
        rocm_target=str((RELEASE / "rocm").resolve()),
        previous_target=str(Path("/opt/bc250-stack/previous").resolve()),
        disk_available_bytes=os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize,
        running_agent_jobs=[],
    )
    atomic_json(LOGS / "final.json", report)
    print(
        "SPARSE_N213_FINAL_PASS",
        len(gates),
        len(report["protected_sha256"]),
        changed,
        flush=True,
    )


if __name__ == "__main__":
    main()
