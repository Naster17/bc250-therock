#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Qualify CPU image codecs and retain video/audio/vision/core coverage."""

import json
from pathlib import Path

from deploy_torch_a04_on_host import sha
from qualify_training_n33_on_host import WS, main
from qualify_training_n39_on_host import preflight

GATES = (
    ("image", "validate_image_b045.py", [], 86),
    ("video", "validate_video_b044.py", [], 55),
    ("audio", "validate_codec_b043.py", [], 46),
    ("vision", "validate_vision_b041.py", ["--group", "integration"], 9),
    ("core", "validate_cleanup_runtime.py", [], 7),
)
SOURCES = (
    "validate_image_b045.py",
    "validate_video_b044.py",
    "validate_codec_b043.py",
    "validate_vision_b041.py",
    "validate_cleanup_runtime.py",
)

if __name__ == "__main__":
    preflight("image-b045")
    stage = json.loads((WS / "logs/image-b045-stage-r1/final.json").read_text())
    assert stage["status"] == "passed"
    candidate = Path(stage["candidate_release"])
    assert sha(candidate / "runtime.json") == stage["candidate_runtime_sha256"]
    main(
        batch_id="image-b045",
        baseline_label="video-b044-qualification-r3",
        include_training_app=False,
        default_phase="extra",
        release_override=candidate,
        extra_gates=GATES,
        additional_sources=SOURCES,
        minimum_disk_free_bytes=4 * 1024**3,
    )
