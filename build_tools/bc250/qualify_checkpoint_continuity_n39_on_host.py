#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Gate historical bound-checkpoint refusal after the candidate CLI batch."""

from pathlib import Path
from qualify_training_n33_on_host import WS, main

if __name__ == "__main__":
    scratch = WS / "experiments/training-n39-r1"
    main(
        batch_id="training-n39",
        baseline_label="training-n39-qualification-r1",
        default_phase="extra",
        extra_gates=(
            (
                "continuity",
                "validate_checkpoint_continuity_n39.py",
                ["--scratch-root", str(scratch)],
                1,
            ),
        ),
        additional_sources=(
            "validate_checkpoint_continuity_n39.py",
            "n37_app_train.py",
            "bc250_checkpoint_provenance.py",
        ),
        scratch_mounts=(scratch, Path("/tmp/bc250-training-n37-r1")),
    )
