#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run vocabulary layout/reference gates separately on the preserved native stack."""

import argparse
import sys

from qualify_training_n33_on_host import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--stage",
        choices=(
            "layout",
            "forward",
            "input-backward",
            "policy-forward",
            "policy-backward",
            "model",
        ),
        required=True,
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    if args.stage == "layout":
        gates = (("layout", "audit_vocab_n38.py", [], 2),)
    elif args.stage == "model":
        gates = (("model", "validate_model_vocab_n38.py", [], 4),)
    else:
        role = (
            "forward"
            if args.stage in ("forward", "policy-forward")
            else "input-backward"
        )
        arguments = ["--role", role]
        if args.stage.startswith("policy-"):
            arguments.append("--match-torch")
        gates = ((args.stage, "benchmark_vocab_n38.py", arguments, 1),)
    main(
        batch_id="training-n38",
        baseline_label="training-n38-entry-r2",
        default_phase="extra",
        extra_gates=gates,
        additional_sources=(
            "audit_vocab_n38.py",
            "benchmark_vocab_n38.py",
            "n38_qualified_bc250_ops.py",
            *(
                (
                    "validate_model_vocab_n38.py",
                    "validate_model_gemm_n32.py",
                    "rocblas_gemm_n38.csv",
                )
                if args.stage == "model"
                else ()
            ),
        ),
        minimum_disk_free_bytes=6 * 1024**3 if args.stage == "layout" else 2 * 1024**3,
    )
