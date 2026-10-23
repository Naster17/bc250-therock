#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Broader stack regressions on the unchanged isolated B03.5 Torch candidate."""

import json
from pathlib import Path

from deploy_torch_a04_on_host import sha
from qualify_distributed_b031_on_host import GATES as RETAINED, SOURCES
from qualify_training_n33_on_host import WS, main
from qualify_training_n39_on_host import preflight

GATES = (
    ("contracts", "validate_rccl_b036.py", [], 178),
    ("torch-rccl", "validate_rccl_b035.py", [], 64),
    ("gpu-gloo", "validate_distributed_b032.py", [], 41),
) + tuple(
    (name, script, ["--nccl-available"] if name == "distributed" else args, count)
    for name, script, args, count in RETAINED
)

if __name__ == "__main__":
    preflight("rccl-b036-qualification")
    staged = json.loads((WS / "logs/rccl-b035-stage-r3/final.json").read_text())
    candidate = Path(staged["candidate_release"])
    assert sha(candidate / "runtime.json") == staged["candidate_runtime_sha256"]
    main(
        batch_id="rccl-b036",
        baseline_label="rccl-b035-qualification-r3",
        include_training_app=False,
        default_phase="extra",
        release_override=candidate,
        extra_gates=GATES,
        scratch_mounts=(WS / "rccl-b034",),
        additional_sources=tuple(sorted(set(SOURCES) | {g[1] for g in GATES})),
        minimum_disk_free_bytes=4 * 1024**3,
    )
