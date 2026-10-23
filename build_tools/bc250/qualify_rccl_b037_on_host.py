#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Qualify new native FP8 guards and retained consumers on an independent candidate."""

import argparse
import json
import re
from pathlib import Path

from deploy_torch_a04_on_host import sha
from qualify_rccl_b036_on_host import GATES as RETAINED
from qualify_distributed_b031_on_host import SOURCES
from qualify_training_n33_on_host import WS, main
from qualify_training_n39_on_host import preflight

GATES = (("guards", "validate_rccl_b037.py", [], 76),) + RETAINED

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--attempt", default="r1")
    args, _ = parser.parse_known_args()
    assert re.fullmatch(r"r[1-9][0-9]*", args.attempt)
    preflight("rccl-b037-qualification")
    stage = json.loads(
        (WS / "logs" / ("rccl-b037-stage-" + args.attempt) / "final.json").read_text()
    )
    candidate = Path(stage["candidate_release"])
    assert sha(candidate / "runtime.json") == stage["candidate_runtime_sha256"]
    main(
        batch_id="rccl-b037",
        baseline_label="rccl-b037-baseline-" + args.attempt,
        include_training_app=False,
        default_phase="extra",
        release_override=candidate,
        extra_gates=GATES,
        scratch_mounts=(WS / "rccl-b034",),
        additional_sources=tuple(sorted(set(SOURCES) | {g[1] for g in GATES})),
        minimum_disk_free_bytes=4 * 1024**3,
    )
