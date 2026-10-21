#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Qualify CPU video references and explicit HIP transfer under prefix lock."""

import json
from pathlib import Path
import sys

from deploy_torch_a04_on_host import sha
from qualify_training_n33_on_host import WS, main
from qualify_training_n39_on_host import preflight

GATES = (
    ("video", "validate_video_b044.py", [], 55),
    ("audio", "validate_codec_b043.py", [], 46),
    ("core", "validate_cleanup_runtime.py", [], 7),
)
SOURCES = (
    "validate_video_b044.py",
    "validate_codec_b043.py",
    "validate_cleanup_runtime.py",
)

if __name__ == "__main__":
    preflight("video-b044")
    candidate = None
    if "--candidate" in sys.argv:
        sys.argv.remove("--candidate")
        stage = json.loads((WS / "logs/video-b044-stage-r1/final.json").read_text())
        assert stage["status"] == "passed"
        candidate = Path(stage["candidate_release"])
        assert sha(candidate / "runtime.json") == stage["candidate_runtime_sha256"]
    main(
        batch_id="video-b044",
        baseline_label="codec-b043-qualification-r3",
        include_training_app=False,
        default_phase="extra",
        release_override=candidate,
        extra_gates=GATES,
        additional_sources=SOURCES,
        minimum_disk_free_bytes=4 * 1024**3,
    )
