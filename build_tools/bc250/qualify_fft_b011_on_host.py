#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Qualify the explicit BF16 FFT adapter and retained native FFT/core gates."""

import json
from pathlib import Path
from deploy_torch_a04_on_host import sha
from qualify_training_n33_on_host import WS, main
from qualify_training_n39_on_host import preflight

GATES = (
    ("promotion", "validate_fft_b011.py", [], 245),
    ("real", "validate_fft_n214.py", ["--group", "real"], 72),
    ("contracts", "validate_fft_n214.py", ["--group", "contracts"], 3),
    ("core", "validate_cleanup_runtime.py", [], 7),
)
SOURCES = (
    "validate_fft_b011.py",
    "validate_fft_n214.py",
    "validate_cleanup_runtime.py",
)

if __name__ == "__main__":
    preflight("fft-b011")
    stage = json.loads((WS / "logs/fft-b011-stage-r3/final.json").read_text())
    assert stage["status"] == "passed"
    candidate = Path(stage["candidate_release"])
    assert sha(candidate / "runtime.json") == stage["candidate_runtime_sha256"]
    main(
        batch_id="fft-b011",
        baseline_label="image-b045-qualification-r3",
        include_training_app=False,
        default_phase="extra",
        release_override=candidate,
        extra_gates=GATES,
        additional_sources=SOURCES,
        minimum_disk_free_bytes=4 * 1024**3,
    )
