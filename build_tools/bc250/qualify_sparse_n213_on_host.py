#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded sparse triangular qualification and reversible exclusive-lock install."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from deploy_torch_a04_on_host import atomic_json, check_gate, sha, switch

WS = Path("/home/nik/bc250work/rocm-10.2-gfx1013")
RELEASE = Path("/opt/bc250-stack/current").resolve()
OLD = WS / "pytorch-profiler-order"
CANDIDATE = WS / "pytorch-sparse-n213"
LOGS = WS / "logs/sparse-n213-qualification"


def protected() -> dict:
    baseline = json.loads(
        (WS / "logs/profiler-order-qualification/final.json").read_text()
    )
    rows = dict(baseline["protected_sha256"])
    rows.update({str(OLD / p): d for p, d in baseline["candidate_sha256"].items()})
    rows["/opt/bc250-stack/current/runtime.json"] = baseline["runtime_sha256"]
    for name in ("SparseBlasImpl.h", "SparseBlas.cpp", "SparseCsrTensorMath.cu"):
        p = OLD / "aten/src/ATen/native/sparse/cuda" / name
        rows[str(p)] = sha(p)
    for p, d in rows.items():
        assert sha(Path(p)) == d, p
    return rows


def identities() -> dict:
    paths = list((CANDIDATE / "torch/lib").glob("*.so")) + list(
        (CANDIDATE / "torch").glob("_C.*.so")
    )
    paths += [
        CANDIDATE / "GIT_REVISION",
        CANDIDATE / "test/test_bc250_sparse_triangular.py",
    ]
    paths += [
        CANDIDATE / "aten/src/ATen/native/sparse/cuda" / n
        for n in ("SparseBlasImpl.h", "SparseBlas.cpp", "SparseCsrTensorMath.cu")
    ]
    return {str(p.relative_to(CANDIDATE)): sha(p) for p in paths}


def gates(phase: str) -> list:
    specs = [
        (g, "validate_sparse_n213.py", ["--group", g])
        for g in ("matrix", "edge", "extended", "unit", "trace")
    ]
    specs += [
        ("native", "validate_sparse_n213_native.py", []),
        ("legacy", "validate_priority2_math.py", ["--group", "sparse"]),
        ("core", "validate_cleanup_runtime.py", []),
        ("profiler", "validate_profiler_order.py", ["--order", "cpu-first"]),
    ]
    expected = {
        "matrix": 256,
        "edge": 80,
        "extended": 32,
        "unit": 64,
        "trace": 1,
        "native": 96,
        "core": 7,
    }
    passed = []
    for name, script, arguments in specs:
        label = f"sparse-n213-{phase}-{name}"
        if name == "edge":
            label += "-r2"
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
        if not (WS / "logs" / label).exists():
            with (LOGS / (label + ".log")).open("w") as output:
                subprocess.run(
                    command,
                    check=True,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=570,
                )
        check_gate(label)
        result = json.loads((WS / "logs" / label / "result.json").read_text())
        assert result["status"] == "passed", label
        if name in expected:
            assert len(result["checks"]) == expected[name], (
                name,
                len(result["checks"]),
            )
        passed.append(label)
        atomic_json(
            LOGS / (phase + "-progress.json"), dict(status="running", passed=passed)
        )
    return passed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=("candidate", "install"), required=True)
    args = ap.parse_args()
    assert (
        os.geteuid() == 0
        and not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    )
    assert (
        json.loads((WS / "logs/sparse-n213-build/exit.json").read_text())["returncode"]
        == 0
    )
    source = json.loads((WS / "helpers/sparse-n213-source.json").read_text())
    assert (CANDIDATE / "GIT_REVISION").read_text().strip() == source["torch_revision"]
    for name, digest in source["helper_sha256"].items():
        assert sha(WS / "helpers" / name) == digest, name
    for name, digest in source["patched_source_sha256"].items():
        assert sha(CANDIDATE / name) == digest, name
    assert sha(CANDIDATE / "torch/lib/libtorch_hip.so") != sha(
        OLD / "torch/lib/libtorch_hip.so"
    )
    assert (RELEASE / "torch").resolve() == OLD
    frozen = protected()
    current = identities()
    LOGS.mkdir(exist_ok=True)
    if args.phase == "candidate":
        passed = gates("candidate")
        assert identities() == current and protected() == frozen
        atomic_json(
            LOGS / "candidate.json",
            dict(
                status="passed",
                source=source,
                candidate_sha256=current,
                protected_sha256=frozen,
                passed_gates=passed,
            ),
        )
        print("SPARSE_N213_CANDIDATE_PASS", len(frozen), flush=True)
        return
    qualified = json.loads((LOGS / "candidate.json").read_text())
    assert (
        qualified["status"] == "passed"
        and qualified["source"] == source
        and qualified["candidate_sha256"] == current
    )
    for label in qualified["passed_gates"]:
        check_gate(label)
    recovery = WS / "recovery/sparse-n213-before"
    recovery.mkdir()
    old_target = os.readlink(RELEASE / "torch")
    old_runtime = (RELEASE / "runtime.json").read_bytes()
    backup = recovery / "runtime.json"
    backup.write_bytes(old_runtime)
    (recovery / "torch-link.txt").write_text(old_target + "\n")
    (recovery / "torch").symlink_to(OLD)
    report = dict(
        status="staging",
        source=source,
        candidate_sha256=current,
        protected_before=frozen,
        old_torch_target=old_target,
        new_torch_target=str(CANDIDATE),
    )
    changed = False
    try:
        switch(CANDIDATE)
        changed = True
        runtime = json.loads(old_runtime)
        runtime["previous_sparse_n213_torch_revision"] = runtime["torch_revision"]
        runtime["torch_revision"] = source["torch_revision"]
        runtime["effective_therock_port_revision"] = source["therock_revision"]
        runtime[
            "source_state"
        ] += "; exact zero/missing sparse triangular pivot rows marked NaN on ROCm"
        runtime["sparse_triangular_n213"] = dict(
            status="installed checks pending",
            source_manifest=str(WS / "SOURCE_SPARSE_N213.json"),
        )
        atomic_json(RELEASE / "runtime.json", runtime)
        report["installed_gates"] = gates("installed")
        assert identities() == current
        preserved = dict(frozen)
        preserved[str(backup)] = preserved.pop("/opt/bc250-stack/current/runtime.json")
        assert all(sha(Path(p)) == d for p, d in preserved.items())
        runtime["sparse_triangular_n213"].update(
            status="bounded generic/native/reference/pivot and retained regression gates passed",
            candidate_sha256=current,
        )
        runtime["profiler_a09"]["library_sha256"] = {
            Path(p).name: d for p, d in current.items() if p.startswith("torch/lib/")
        }
        runtime["profiler_a09"]["source_manifest"] = str(WS / "SOURCE_SPARSE_N213.json")
        atomic_json(RELEASE / "runtime.json", runtime)
        report.update(
            status="passed",
            protected_after=preserved,
            protected_unchanged_except_saved_runtime=True,
            kernel=os.uname().release,
            boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            disk_available_bytes=os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize,
            running_agent_jobs=[],
        )
        for path in (
            WS / "SOURCE_SPARSE_N213.json",
            RELEASE / "SOURCE_SPARSE_N213.json",
            LOGS / "installed.json",
        ):
            atomic_json(path, report)
        print("SPARSE_N213_INSTALLED_PASS", len(preserved), flush=True)
    except BaseException as error:
        if changed:
            switch(old_target)
            temporary = RELEASE / "runtime.sparse-n213-rollback"
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
