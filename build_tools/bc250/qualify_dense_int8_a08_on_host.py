#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run bounded integrated gates; restore the paired old Lt on failure."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from deploy_torch_a04_on_host import atomic_json, check_gate

WS = Path("/home/nik/bc250work/rocm-10.2-gfx1013")


def main() -> None:
    assert os.geteuid() == 0
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    check_gate("dense-int8-a08-candidate-r2", 16)
    check_gate("dense-int8-a08-float-regressions", 3)
    logs = WS / "logs/dense-int8-a08-qualification"
    logs.mkdir()
    report = dict(
        status="running",
        passed_gates=[
            "dense-int8-a08-candidate-r2",
            "dense-int8-a08-float-regressions",
        ],
    )
    try:
        for suffix, script, args, count in [
            ("installed", "validate_dense_int8_a08.py", [], 16),
            ("core", "validate_cleanup_runtime.py", [], None),
            ("profiler", "validate_profiler_a09.py", [], 7),
            ("sparse", "validate_sparse_followup.py", [], 38),
            ("conv", "validate_miopen_a01_torch.py", [], 30),
            ("a06", "validate_a06_a07.py", ["--component=a06"], 42),
            ("a07", "validate_a06_a07.py", ["--component=a07"], 36),
            ("model", "validate_a06_a07_model.py", [], 3),
        ]:
            label = "dense-int8-a08-" + suffix
            report["phase"] = label
            atomic_json(logs / "result.json", report)
            command = [
                sys.executable,
                str(WS / "helpers/validate_torch_a04_on_host.py"),
                "--label",
                label,
                "--script",
                script,
            ] + ["--test-arg=" + a for a in args]
            with (logs / (suffix + ".log")).open("w") as output:
                subprocess.run(
                    command,
                    check=True,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=570,
                )
            check_gate(label)
            result = json.loads((WS / "logs" / label / "result.json").read_text())
            assert result["status"] == "passed"
            if count is not None:
                assert len(result["results"]) == count
            report["passed_gates"].append(label)
        report["status"] = "passed"
        atomic_json(logs / "result.json", report)
    except BaseException as error:
        recovery = WS / "recovery/dense-int8-a08-before"
        for old, new in [
            (
                "gfx1013_scalar.cpp",
                "source/rocm-libraries/projects/hipblaslt/library/src/amd_detail/gfx1013_scalar.cpp",
            ),
            ("libhipblaslt.so.1.5", "build/dist/rocm/lib/libhipblaslt.so.1.5"),
            ("blas_lib_gfx1013.kpack", "build/dist/rocm/.kpack/blas_lib_gfx1013.kpack"),
        ]:
            shutil.copy2(recovery / old, WS / new)
        report.update(status="failed_runtime_rolled_back", error=str(error))
        atomic_json(logs / "result.json", report)
        raise


if __name__ == "__main__":
    main()
