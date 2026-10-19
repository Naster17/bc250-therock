#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Installed automatic preset: one model worker against the qualified baseline."""
import argparse
import json
import os
from pathlib import Path
import torch
from validate_model_gemm_n32 import worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert (
        os.environ.get("ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH")
        == "/workspace/rocblas-gemm-overrides.csv"
    )
    assert "LD_PRELOAD" not in os.environ
    args.worker = "candidate"
    worker(args)
    result = json.loads(args.output.read_text())
    root = args.output.parent.parent / "gemm-n32-model-candidate-r1"
    ref = json.loads((root / "worker-0-reference.json").read_text())
    reference = torch.load(
        root / ref["gradients"], weights_only=True, map_location="cpu"
    )
    gradients = torch.load(
        args.output.with_suffix(".gradients.pt"), weights_only=True, map_location="cpu"
    )
    assert result["loss"] == ref["loss"] and gradients.keys() == reference.keys()
    errors = {
        n: float((g - reference[n]).norm() / reference[n].norm().clamp_min(1e-12))
        for n, g in gradients.items()
    }
    assert max(errors.values()) < 0.04
    assert result["cpu_rng_sha256"] == ref["cpu_rng_sha256"]
    assert result["gpu_rng_sha256"] == ref["gpu_rng_sha256"]
    result.update(
        checks=[
            "automatic-preset",
            "372-reference-gradients",
            "cpu-adamw-update-state",
            "rng-replay",
        ],
        max_adapter_gradient_relative_l2=max(errors.values()),
        reference_wall_ratio=ref["median_wall_seconds"] / result["median_wall_seconds"],
    )
    assert result["reference_wall_ratio"] > 1.1
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
