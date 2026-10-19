#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Seal a bounded exact-output candidate audit after the first model rejection."""
from qualify_training_n33_on_host import main


if __name__ == "__main__":
    main(
        batch_id="training-n34",
        shape=(1024, 2048, 2048),
        baseline_label="training-n33-qualification-r4",
        default_phase="extra",
        extra_gates=(("equivalence", "validate_gemm_equivalence_n34.py", [], 1),),
        additional_sources=("validate_gemm_equivalence_n34.py",),
    )
