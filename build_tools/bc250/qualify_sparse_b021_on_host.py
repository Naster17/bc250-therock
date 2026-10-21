#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Qualify the bounded sparse solve adapter and retained sparse/core gates."""

import json
from pathlib import Path
from deploy_torch_a04_on_host import sha
from qualify_training_n33_on_host import WS, main
from qualify_training_n39_on_host import preflight

GATES = (
    ("general", "validate_sparse_b021.py", [], 253),
    ("edge", "validate_sparse_n213.py", ["--group", "edge"], 80),
    ("legacy", "validate_priority2_math.py", ["--group", "sparse"], 41),
    ("core", "validate_cleanup_runtime.py", [], 7),
)
SOURCES = (
    "validate_sparse_b021.py",
    "validate_sparse_n213.py",
    "test_bc250_sparse_triangular.py",
    "validate_priority2_math.py",
    "validate_cleanup_runtime.py",
)

if __name__ == "__main__":
    preflight("sparse-b021")
    stage = json.loads((WS / "logs/sparse-b021-stage-r4/final.json").read_text())
    assert stage["status"] == "passed"
    candidate = Path(stage["candidate_release"])
    assert sha(candidate / "runtime.json") == stage["candidate_runtime_sha256"]
    main(
        batch_id="sparse-b021",
        baseline_label="fft-b011-qualification-r3",
        include_training_app=False,
        default_phase="extra",
        release_override=candidate,
        extra_gates=GATES,
        additional_sources=SOURCES,
        minimum_disk_free_bytes=4 * 1024**3,
    )
