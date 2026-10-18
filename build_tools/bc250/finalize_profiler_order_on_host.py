#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Seal profiler-order deployment identities, all guards and retained fallbacks."""

import json
import os
from pathlib import Path
import subprocess

from deploy_torch_a04_on_host import atomic_json, check_gate, sha
from qualify_profiler_order_on_host import CANDIDATE, LOGS, OLD, RELEASE, WS


def main() -> None:
    assert os.geteuid() == 0
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    installed = json.loads((LOGS / "installed.json").read_text())
    candidate = json.loads((LOGS / "candidate.json").read_text())
    assert installed["status"] == candidate["status"] == "passed"
    assert (RELEASE / "torch").resolve() == CANDIDATE
    assert (
        json.loads((RELEASE / "runtime.json").read_text())["torch_revision"]
        == installed["source"]["torch_revision"]
    )
    assert len(installed["protected_after"]) == 104
    for p, d in installed["protected_after"].items():
        assert sha(Path(p)) == d, p
    for p, d in installed["candidate_sha256"].items():
        assert sha(CANDIDATE / p) == d, p
    # Qualified test helpers and deployed tests are the same final revision.
    assert sha(CANDIDATE / "test/test_bc250_profiler_order.py") == sha(
        WS / "helpers/test_bc250_profiler_order.py"
    )
    for label in candidate["passed_gates"] + installed["installed_gates"]:
        check_gate(label)
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    failures = {
        "profiler-order-before-gpu-first",
        "profiler-order-candidate-masked",
        "profiler-order-candidate-masked-r2",
    }
    gates = []
    for folder in sorted((WS / "logs").glob("profiler-order-*")):
        if not (folder / "guard.log").exists():
            continue
        status = json.loads((folder / "exit.json").read_text())
        assert status == dict(
            returncode=int(folder.name in failures), kernel_log_unchanged=True
        ), folder
        state = json.loads((Path("/run") / folder.name / "state.json").read_text())
        assert (
            state["status"] == "acknowledged"
            and state["timeout_seconds"] == 600
            and state["boot_id"] == boot
        )
        gates.append(dict(label=folder.name, **status, guard_status="acknowledged"))
    assert len(gates) == 21 and sum(g["returncode"] != 0 for g in gates) == 3
    assert (
        subprocess.check_output(["dmesg"])
        == (WS / "logs/profiler-order-before-cpu-first/kernel-before.log").read_bytes()
    )
    changed = [
        p.name
        for p in (CANDIDATE / "torch/lib").glob("*.so")
        if sha(p) != sha(OLD / "torch/lib" / p.name)
    ]
    assert "libtorch_cpu.so" in changed and "libtorch_hip.so" not in changed
    names = [
        "build_profiler_order_on_host.py",
        "build_torch_profiler_order.sh",
        "qualify_profiler_order_on_host.py",
        "validate_profiler_order.py",
        "test_bc250_profiler_order.py",
        "finalize_profiler_order_on_host.py",
    ]
    report = dict(
        status="passed",
        native_torch_revision=installed["source"]["torch_revision"],
        build_therock_revision=installed["source"]["therock_revision"],
        source_sha256={name: sha(WS / "helpers" / name) for name in names},
        candidate_sha256=installed["candidate_sha256"],
        protected_sha256=installed["protected_after"],
        gates=gates,
        changed_native_libraries=changed,
        gpu_arithmetic_library_unchanged=True,
        kernel_log_unchanged_entire_batch=True,
        boot_id=boot,
        kernel=os.uname().release,
        torch_target=str((RELEASE / "torch").resolve()),
        rocm_target=str((RELEASE / "rocm").resolve()),
        previous_target=str(Path("/opt/bc250-stack/previous").resolve()),
        runtime_sha256=sha(RELEASE / "runtime.json"),
        disk_available_bytes=os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize,
        running_agent_jobs=[],
    )
    atomic_json(LOGS / "final.json", report)
    print(
        "PROFILER_ORDER_FINAL_PASS",
        len(gates),
        len(report["protected_sha256"]),
        changed,
        flush=True,
    )


if __name__ == "__main__":
    main()
