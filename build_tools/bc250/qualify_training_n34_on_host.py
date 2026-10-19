#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit square BF16 projection layouts on the unchanged qualified stack."""
from qualify_training_n33_on_host import main


if __name__ == "__main__":
    main(
        batch_id="training-n34",
        shape=(1024, 2048, 2048),
        baseline_label="training-n33-qualification-r4",
        default_phase="gemm",
        extra_gates=(("model", "validate_model_gemm_n34.py", [], 4),),
        additional_sources=(
            "validate_model_gemm_n34.py",
            "validate_model_gemm_n32.py",
            "rocblas_gemm_n34.csv",
        ),
    )
