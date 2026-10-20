#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Guarded broad vision GPU/reference gates on the isolated candidate image."""

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
    preflight("vision-b041-" + args.attempt)
    shutil.copy2(
        WS / "logs" / ("training-n39-resource-vision-b041-" + args.attempt + ".json"),
        WS / "logs" / ("vision-b041-resource-" + args.attempt + ".json"),
    )
    stage = json.loads((WS / "logs/vision-b041-stage-r2/final.json").read_text())
    candidate = Path(stage["candidate_release"])
    assert sha(candidate / "runtime.json") == stage["candidate_runtime_sha256"]
    main(
        batch_id="vision-b041",
        baseline_label="vision-b041-entry-r2",
        default_phase="extra",
        release_override=candidate,
        extra_gates=(
            ("ops", "validate_vision_b041.py", ["--group", "ops"], 51),
            ("integration", "validate_vision_b041.py", ["--group", "integration"], 9),
            ("core", "validate_cleanup_runtime.py", [], 7),
        ),
        additional_sources=("validate_vision_b041.py", "validate_cleanup_runtime.py"),
        minimum_disk_free_bytes=4 * 1024**3,
    )
