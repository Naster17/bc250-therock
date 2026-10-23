#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Qualify an isolated NCCL-enabled Torch candidate on one physical gfx1013."""

import argparse
import json
import re
from pathlib import Path

from deploy_torch_a04_on_host import sha
from qualify_training_n33_on_host import WS, main
from qualify_training_n39_on_host import preflight

GATES = (
    ("torch-rccl", "validate_rccl_b035.py", [], 64),
    ("gpu-gloo", "validate_distributed_b032.py", [], 41),
    ("cpu-gloo", "validate_distributed_b031.py", ["--nccl-available"], 31),
    ("compiled-dense", "validate_compiler_n310.py", ["--family", "dense"], 14),
    ("compiled-cnn", "validate_compiler_n310.py", ["--family", "cnn"], 14),
    ("profiler", "validate_distributed_profiler_b031.py", [], 1),
    ("core", "validate_cleanup_runtime.py", [], 7),
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--attempt", default="r1")
    args, _ = parser.parse_known_args()
    assert re.fullmatch(r"r[1-9][0-9]*", args.attempt)
    preflight("rccl-b035-qualification")
    stage = json.loads(
        (WS / "logs" / ("rccl-b035-stage-" + args.attempt) / "final.json").read_text()
    )
    candidate = Path(stage["candidate_release"])
    assert sha(candidate / "runtime.json") == stage["candidate_runtime_sha256"]
    main(
        batch_id="rccl-b035",
        baseline_label="rccl-b035-baseline-" + args.attempt,
        include_training_app=False,
        default_phase="extra",
        release_override=candidate,
        extra_gates=GATES,
        scratch_mounts=(WS / "rccl-b034",),
        additional_sources=tuple(
            sorted(
                {g[1] for g in GATES}
                | {"validate_profiler_order.py", "test_bc250_profiler_order.py"}
            )
        ),
        minimum_disk_free_bytes=4 * 1024**3,
    )
