#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare vocabulary candidates against the preserved N3.2 model baseline."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from validate_model_gemm_n32 import worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=("reference", "candidate"))
    args = parser.parse_args()
    if args.worker:
        return worker(
            args, app_path=Path(__file__).with_name("n38_qualified_bc250_ops.py")
        )
    baseline = Path("/workspace/rocblas-gemm-overrides.csv")
    candidate = Path(__file__).with_name("rocblas_gemm_n38.csv")
    assert os.environ.get("ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH") == str(baseline)
    assert "LD_PRELOAD" not in os.environ
    original = baseline.read_text().splitlines()
    extended = candidate.read_text().splitlines()
    assert len(original) == 2 and extended[:2] == original and len(extended) in (3, 4)
    report = dict(
        status="running",
        checks=[],
        results=[],
        baseline_csv_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
        candidate_csv_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
        contract="Existing model abs loss <0.015 and max adapter gradient relativeL2 <0.04; each worker CPU AdamW replay unchanged; reference repetition exact",
        performance="Three warm complete steps per fresh ABBA worker; no speed gate or deployment implied",
    )
    reference = None
    reference_gradients = None
    try:
        for index, mode in enumerate(
            ("reference", "candidate", "candidate", "reference")
        ):
            output = args.output.parent / f"worker-{index}-{mode}.json"
            env = dict(os.environ)
            env["ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH"] = str(
                baseline if mode == "reference" else candidate
            )
            command = [
                sys.executable,
                str(Path(__file__)),
                "--worker",
                mode,
                "--output",
                str(output),
            ]
            with output.with_suffix(".log").open("w") as log:
                subprocess.run(
                    command,
                    env=env,
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=150,
                )
            result = json.loads(output.read_text())
            assert result["status"] == "passed" and result["gradient_count"] == 372
            gradients = torch.load(
                output.with_suffix(".gradients.pt"),
                weights_only=True,
                map_location="cpu",
            )
            if reference is None:
                reference, reference_gradients = result, gradients
            else:
                loss_error = abs(result["loss"] - reference["loss"])
                assert loss_error < 0.015, loss_error
                assert gradients.keys() == reference_gradients.keys()
                errors = {
                    n: float(
                        (g - reference_gradients[n]).norm()
                        / reference_gradients[n].norm().clamp_min(1e-12)
                    )
                    for n, g in gradients.items()
                }
                assert max(errors.values()) < 0.04, max(errors.values())
                exact = all(
                    torch.equal(gradients[n], reference_gradients[n]) for n in gradients
                )
                if mode == "reference":
                    assert exact and loss_error == 0
                assert result["cpu_rng_sha256"] == reference["cpu_rng_sha256"]
                assert result["gpu_rng_sha256"] == reference["gpu_rng_sha256"]
                result.update(
                    max_adapter_gradient_relative_l2=max(errors.values()),
                    loss_abs_error=loss_error,
                    exact_gradients=exact,
                )
            report["results"].append(result)
            report["checks"].append(
                f"worker-{index}-{mode}-372-gradients-cpu-adamw-rng"
            )
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        report["paired_wall_ratios"] = [
            report["results"][0]["median_wall_seconds"]
            / report["results"][1]["median_wall_seconds"],
            report["results"][3]["median_wall_seconds"]
            / report["results"][2]["median_wall_seconds"],
        ]
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
