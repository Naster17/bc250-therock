#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Guarded broad audio GPU/reference gates on the isolated candidate image."""

import argparse
import json
from pathlib import Path
import re
import shutil

from deploy_torch_a04_on_host import sha
from qualify_training_n33_on_host import WS, main
from qualify_training_n39_on_host import preflight

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--attempt", default="r1")
    args, _ = parser.parse_known_args()
    assert re.fullmatch(r"r[1-9][0-9]*", args.attempt)
    preflight("audio-b042-" + args.attempt)
    shutil.copy2(
        WS / "logs" / ("training-n39-resource-audio-b042-" + args.attempt + ".json"),
        WS / "logs" / ("audio-b042-resource-" + args.attempt + ".json"),
    )
    stage = json.loads((WS / "logs/audio-b042-stage-r2/final.json").read_text())
    candidate = Path(stage["candidate_release"])
    assert sha(candidate / "runtime.json") == stage["candidate_runtime_sha256"]
    main(
        batch_id="audio-b042",
        baseline_label="audio-b042-entry-r1",
        default_phase="extra",
        release_override=candidate,
        hip_candidate=WS
        / "audio-b042/runtime-candidate-r1/libamdhip64.so.7.17.0-0000000",
        extra_gates=(
            ("dsp", "validate_audio_b042.py", ["--group", "dsp"], 34),
            ("native", "validate_audio_b042.py", ["--group", "native"], 40),
            ("vision-ops", "validate_vision_b041.py", ["--group", "ops"], 51),
            (
                "vision-integration",
                "validate_vision_b041.py",
                ["--group", "integration"],
                9,
            ),
            ("core", "validate_cleanup_runtime.py", [], 7),
        ),
        additional_sources=(
            "validate_audio_b042.py",
            "validate_vision_b041.py",
            "validate_cleanup_runtime.py",
        ),
        minimum_disk_free_bytes=4 * 1024**3,
    )
