#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Single-device native RCCL references plus retained Gloo/core consumers."""

from qualify_training_n33_on_host import WS, main
from qualify_training_n39_on_host import preflight

GATES = (
    ("native", "validate_rccl_b033.py", [], 82),
    ("gpu-gloo", "validate_distributed_b032.py", [], 41),
    ("cpu-gloo", "validate_distributed_b031.py", [], 31),
    ("core", "validate_cleanup_runtime.py", [], 7),
)

if __name__ == "__main__":
    preflight("rccl-b033-qualification")
    main(
        batch_id="rccl-b033",
        baseline_label="rccl-b033-baseline-r1",
        include_training_app=False,
        default_phase="extra",
        extra_gates=GATES,
        scratch_mounts=(WS / "rccl-b033",),
        additional_sources=(
            "validate_rccl_b033.py",
            "rccl_readiness_b033.cpp",
            "validate_distributed_b032.py",
            "validate_distributed_b031.py",
            "validate_cleanup_runtime.py",
        ),
        minimum_disk_free_bytes=4 * 1024**3,
    )
