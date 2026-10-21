#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Recheck installed native audio/core/hostcall after an observed new boot."""

from qualify_training_n33_on_host import main
from qualify_training_n39_on_host import preflight

preflight("audio-b042-postboot-r6")
main(
    batch_id="audio-b042",
    baseline_label="audio-b042-qualification-r5",
    default_phase="extra",
    extra_gates=(
        ("native", "validate_audio_b042.py", ["--group", "native"], 40),
        ("core", "validate_cleanup_runtime.py", [], 7),
        ("hostcall", "validate_hostcall_queue_b042.py", [], 1),
    ),
    additional_sources=(
        "validate_audio_b042.py",
        "validate_cleanup_runtime.py",
        "validate_hostcall_queue_b042.py",
    ),
    minimum_disk_free_bytes=4 * 1024**3,
)
