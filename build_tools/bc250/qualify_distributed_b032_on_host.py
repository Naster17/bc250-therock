#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded installed Gloo GPU software-transport references on a single board."""

from qualify_training_n33_on_host import main
from qualify_training_n39_on_host import preflight

GATES = (
    ("gpu", "validate_distributed_b032.py", [], 41),
    ("cpu", "validate_distributed_b031.py", [], 31),
    ("compiled-dense", "validate_compiler_n310.py", ["--family", "dense"], 14),
    ("compiled-cnn", "validate_compiler_n310.py", ["--family", "cnn"], 14),
    ("profiler", "validate_distributed_profiler_b031.py", [], 1),
    ("core", "validate_cleanup_runtime.py", [], 7),
)

if __name__ == "__main__":
    preflight("distributed-b032")
    main(
        batch_id="distributed-b032",
        baseline_label="distributed-b031-qualification-r3",
        include_training_app=False,
        default_phase="extra",
        extra_gates=GATES,
        additional_sources=(
            "validate_distributed_b032.py",
            "validate_distributed_b031.py",
            "validate_compiler_n310.py",
            "validate_distributed_profiler_b031.py",
            "validate_profiler_order.py",
            "test_bc250_profiler_order.py",
            "validate_cleanup_runtime.py",
        ),
        minimum_disk_free_bytes=4 * 1024**3,
    )
