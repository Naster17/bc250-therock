#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded profiler-order qualification and reversible exclusive-lock deployment."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from deploy_torch_a04_on_host import atomic_json, check_gate, sha, switch

WS = Path("/home/nik/bc250work/rocm-10.2-gfx1013")
RELEASE = Path("/opt/bc250-stack/current").resolve()
CANDIDATE = WS / "pytorch-profiler-order"
OLD = WS / "pytorch-profiler-a09"
LOGS = WS / "logs/profiler-order-qualification"


def protected() -> dict:
    rows = json.loads((WS / "logs/solver-backsub-n212-r2/final.json").read_text())[
        "protected_sha256"
    ]
    for path, digest in rows.items():
        assert sha(Path(path)) == digest, path
    return rows


def candidate_hashes() -> dict:
    paths = list((CANDIDATE / "torch/lib").glob("*.so")) + list(
        (CANDIDATE / "torch").glob("_C.*.so")
    )
    paths += [
        CANDIDATE / "GIT_REVISION",
        CANDIDATE / "torch/csrc/profiler/kineto_shim.cpp",
        CANDIDATE / "test/test_bc250_profiler_order.py",
    ]
    return {str(p.relative_to(CANDIDATE)): sha(p) for p in paths}


def gates(phase: str) -> list:
    passed = []
    rows = [
        (mode, "validate_profiler_order.py", ["--order", mode])
        for mode in ("cpu-first", "gpu-first", "cpu-cold", "masked")
    ]
    rows += [
        ("api", "validate_profiler_a09.py", []),
        ("core", "validate_cleanup_runtime.py", []),
    ]
    for name, script, arguments in rows:
        label = f"profiler-order-{phase}-{name}"
        command = [
            sys.executable,
            str(WS / "helpers/validate_torch_a04_on_host.py"),
            "--label",
            label,
            "--script",
            script,
        ]
        if phase == "candidate":
            command += ["--torch-candidate", str(CANDIDATE)]
        command += ["--test-arg=" + a for a in arguments]
        with (LOGS / (label + ".log")).open("w") as output:
            subprocess.run(
                command,
                check=True,
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=570,
            )
        check_gate(label)
        report = json.loads((WS / "logs" / label / "result.json").read_text())
        assert report["status"] == "passed", label
        assert not report.get("expected_baseline_defect", False), label
        if name == "api":
            assert len(report["results"]) == 7
        if name == "core":
            assert len(report["checks"]) == 7
        passed.append(label)
        atomic_json(
            LOGS / (phase + "-progress.json"), dict(status="running", passed=passed)
        )
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("candidate", "install"), required=True)
    args = parser.parse_args()
    assert os.geteuid() == 0
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    assert (
        json.loads((WS / "logs/profiler-order-build/exit.json").read_text())[
            "returncode"
        ]
        == 0
    )
    source = json.loads((WS / "helpers/profiler-order-source.json").read_text())
    assert (CANDIDATE / "GIT_REVISION").read_text().strip() == source["torch_revision"]
    assert (
        sha(CANDIDATE / "torch/csrc/profiler/kineto_shim.cpp") == source["shim_sha256"]
    )
    assert sha(CANDIDATE / "test/test_bc250_profiler_order.py") == source["test_sha256"]
    # No GPU arithmetic DSO change is needed for this CPU-side initialization fix.
    assert sha(CANDIDATE / "torch/lib/libtorch_hip.so") == sha(
        OLD / "torch/lib/libtorch_hip.so"
    )
    frozen = protected()
    assert (RELEASE / "torch").resolve() == OLD
    LOGS.mkdir(exist_ok=True)
    identities = candidate_hashes()
    if args.phase == "candidate":
        passed = gates("candidate")
        assert candidate_hashes() == identities
        assert protected() == frozen
        atomic_json(
            LOGS / "candidate.json",
            dict(
                status="passed",
                source=source,
                candidate_sha256=identities,
                protected_sha256=frozen,
                passed_gates=passed,
            ),
        )
        print("PROFILER_ORDER_CANDIDATE_PASS", flush=True)
        return
    qualified = json.loads((LOGS / "candidate.json").read_text())
    assert (
        qualified["status"] == "passed" and qualified["candidate_sha256"] == identities
    )
    assert qualified["source"] == source
    for label in qualified["passed_gates"]:
        check_gate(label)
    old_target = os.readlink(RELEASE / "torch")
    old_runtime = (RELEASE / "runtime.json").read_bytes()
    recovery = WS / "recovery/profiler-order-before"
    recovery.mkdir()
    backup = recovery / "runtime.json"
    backup.write_bytes(old_runtime)
    (recovery / "torch-link.txt").write_text(old_target + "\n")
    report = dict(
        status="staging",
        source=source,
        old_torch_target=old_target,
        new_torch_target=str(CANDIDATE),
        protected_before=frozen,
        candidate_sha256=identities,
    )
    changed = False
    try:
        switch(CANDIDATE)
        changed = True
        runtime = json.loads(old_runtime)
        runtime["previous_profiler_order_torch_revision"] = runtime["torch_revision"]
        runtime["torch_revision"] = source["torch_revision"]
        runtime["effective_therock_port_revision"] = source["therock_revision"]
        runtime[
            "source_state"
        ] += "; ROCm profiler CPU-first initialization-order correction"
        runtime["profiler_initialization_order"] = dict(
            status="installed checks pending",
            source_manifest=str(WS / "SOURCE_PROFILER_ORDER.json"),
            previous_profiler_a09=runtime.get("profiler_a09"),
        )
        atomic_json(RELEASE / "runtime.json", runtime)
        report["installed_gates"] = gates("installed")
        assert candidate_hashes() == identities
        preserved = dict(frozen)
        digest = preserved.pop("/opt/bc250-stack/current/runtime.json")
        preserved[str(backup)] = digest
        assert all(sha(Path(p)) == d for p, d in preserved.items())
        runtime["profiler_initialization_order"].update(
            status="bounded CPU-first/GPU-first/cold/masked and API/core gates passed",
            candidate_sha256=identities,
        )
        runtime["profiler_a09"]["library_sha256"] = {
            Path(p).name: d for p, d in identities.items() if p.startswith("torch/lib/")
        }
        runtime["profiler_a09"]["source_manifest"] = str(
            WS / "SOURCE_PROFILER_ORDER.json"
        )
        atomic_json(RELEASE / "runtime.json", runtime)
        report.update(
            status="passed",
            protected_after=preserved,
            protected_unchanged_except_saved_runtime=True,
            running_agent_jobs=[],
            kernel=os.uname().release,
            boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            disk_available_bytes=os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize,
        )
        for path in (
            WS / "SOURCE_PROFILER_ORDER.json",
            RELEASE / "SOURCE_PROFILER_ORDER.json",
            LOGS / "installed.json",
        ):
            atomic_json(path, report)
        print("PROFILER_ORDER_INSTALLED_PASS", len(preserved), flush=True)
    except BaseException as error:
        if changed:
            switch(old_target)
            temporary = RELEASE / "runtime.profiler-order-rollback"
            assert not temporary.exists()
            temporary.write_bytes(old_runtime)
            os.replace(temporary, RELEASE / "runtime.json")
        report.update(
            status="failed_rolled_back" if changed else "failed_before_switch",
            error=str(error),
        )
        atomic_json(LOGS / "failed.json", report)
        raise


if __name__ == "__main__":
    main()
