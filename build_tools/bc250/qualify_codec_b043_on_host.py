#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Guarded codec reference tests and retained audio/vision/core gates."""

import json
from pathlib import Path

from deploy_torch_a04_on_host import sha
from qualify_training_n33_on_host import WS, main
from qualify_training_n39_on_host import preflight

GATES = (
    ("files", "validate_codec_b043.py", [], 46),
    ("dsp", "validate_audio_b042.py", ["--group", "dsp"], 34),
    (
        "native",
        "validate_audio_b042.py",
        ["--group", "native", "--codec-available"],
        40,
    ),
    ("vision", "validate_vision_b041.py", ["--group", "integration"], 9),
    ("core", "validate_cleanup_runtime.py", [], 7),
)
SOURCES = (
    "validate_codec_b043.py",
    "validate_audio_b042.py",
    "validate_vision_b041.py",
    "validate_cleanup_runtime.py",
)

if __name__ == "__main__":
    preflight("codec-b043-candidate-r1")
    stage = json.loads((WS / "logs/codec-b043-stage-r5/final.json").read_text())
    assert stage["status"] == "passed"
    candidate = Path(stage["candidate_release"])
    assert sha(candidate / "runtime.json") == stage["candidate_runtime_sha256"]
    main(
        batch_id="codec-b043",
        include_training_app=False,
        baseline_label="codec-b043-entry-r4",
        default_phase="extra",
        release_override=candidate,
        extra_gates=GATES,
        additional_sources=SOURCES,
        minimum_disk_free_bytes=4 * 1024**3,
    )
