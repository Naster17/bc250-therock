#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Replay an owned RAM checkpoint without repeating retained accumulation checks."""
from pathlib import Path
from qualify_training_n33_on_host import main


if __name__ == "__main__":
    root = Path("/tmp/bc250-training-n35-r2")
    assert root.is_dir() and (root / "uninterrupted-step2.pt").is_file()
    main(
        batch_id="training-n35",
        baseline_label="training-n34-qualification-r3",
        default_phase="extra",
        extra_gates=(("replay", "replay_training_n35.py", [], 8),),
        additional_sources=("replay_training_n35.py", "validate_training_n35.py"),
        scratch_mounts=(root,),
        minimum_disk_free_bytes=512 * 1024**2,
    )
