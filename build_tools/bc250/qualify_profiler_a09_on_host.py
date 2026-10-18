#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Candidate regression gates under update_dev_prefix.py's exclusive lock."""
import json
import os
from pathlib import Path
import subprocess
import sys

from deploy_torch_a04_on_host import atomic_json, check_gate, sha

WS = Path("/home/nik/bc250work/rocm-10.2-gfx1013")
CANDIDATE = WS / "pytorch-profiler-a09"


def main() -> None:
    assert os.geteuid() == 0
    assert (
        json.loads((WS / "logs/profiler-a09-build/exit.json").read_text())["returncode"]
        == 0
    )
    check_gate("profiler-a09-candidate-r2", 7)
    folder = WS / "logs/profiler-a09-candidate-regressions"
    folder.mkdir()
    report = dict(status="running", passed_gates=["profiler-a09-candidate-r2"])
    for name, script, args, count in [
        ("core", "validate_cleanup_runtime.py", [], None),
        ("sparse", "validate_sparse_followup.py", [], 38),
        ("training", "validate_training_a05.py", ["--installed"], 69),
        ("a06", "validate_a06_a07.py", ["--component=a06"], 42),
        ("a07", "validate_a06_a07.py", ["--component=a07"], 36),
        ("conv", "validate_miopen_a01_torch.py", [], 30),
        ("model", "validate_a06_a07_model.py", [], 3),
        ("cli", "validate_a06_a07_cli.py", [], 2),
    ]:
        label = "profiler-a09-candidate-" + name
        report["phase"] = label
        atomic_json(folder / "result.json", report)
        command = [
            sys.executable,
            str(WS / "helpers/validate_torch_a04_on_host.py"),
            "--label",
            label,
            "--script",
            script,
            "--torch-candidate",
            str(CANDIDATE),
        ]
        command += ["--test-arg=" + a for a in args]
        with (folder / (name + ".log")).open("w") as output:
            subprocess.run(
                command,
                check=True,
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=570,
            )
        check_gate(label, count)
        assert (
            json.loads((WS / "logs" / label / "result.json").read_text())["status"]
            == "passed"
        )
        report["passed_gates"].append(label)
    report.update(
        status="passed",
        candidate_library_sha256={
            p.name: sha(p) for p in (CANDIDATE / "torch/lib").glob("*.so")
        },
    )
    atomic_json(folder / "result.json", report)


if __name__ == "__main__":
    main()
