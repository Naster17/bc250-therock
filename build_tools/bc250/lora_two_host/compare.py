#!/usr/bin/env python3
"""Compare first-update LoRA gradients to an equal-global-batch one-GPU reference.

Predeclared FP16 bounds: whole-vector relative L2 <= 0.03; every parameter
error L2 <= 0.05 * reference L2 + 3e-6 * sqrt(elements). This compares the
same GPU model graph across topologies, not an independent CPU model reference.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def compare(reference: Path, candidate: Path):
    a = json.loads(reference.with_suffix(".json").read_text())
    b = json.loads(candidate.with_suffix(".json").read_text())
    assert a == b
    x = np.memmap(reference, mode="r", dtype="<f4")
    y = np.memmap(candidate, mode="r", dtype="<f4")
    assert x.size == y.size
    error2 = norm2 = 0.0
    worst = 0.0
    present = 0
    for row in a:
        if row.get("absent"):
            continue
        start, count = row["offset"], row["elements"]
        ar = x[start : start + count].astype(np.float64)
        br = y[start : start + count].astype(np.float64)
        assert np.isfinite(ar).all() and np.isfinite(br).all()
        norm = float(np.linalg.norm(ar))
        error = float(np.linalg.norm(br - ar))
        limit = 0.05 * norm + 3e-6 * math.sqrt(count)
        assert error <= limit, (row["name"], error, limit)
        worst = max(worst, error / limit)
        norm2 += norm * norm
        error2 += error * error
        present += 1
    relative = math.sqrt(error2 / norm2)
    assert relative <= 0.03, relative
    return {
        "status": "passed",
        "present_parameter_tensors": present,
        "elements": x.size,
        "relative_l2": relative,
        "maximum_per_parameter_bound_fraction": worst,
        "reference_sha256": hashlib.file_digest(
            reference.open("rb"), "sha256"
        ).hexdigest(),
        "candidate_sha256": hashlib.file_digest(
            candidate.open("rb"), "sha256"
        ).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
