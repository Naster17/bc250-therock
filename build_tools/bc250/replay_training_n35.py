#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Finish the retained accumulation fixture after isolating CPU-oracle state."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch
from profile_training_n31 import identical
from validate_training_n35 import sha


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logs = args.output.parent.parent
    previous = logs / "training-n35-r2-accum-resume"
    qualification = json.loads(
        (logs / "training-n35-qualification-r2/final.json").read_text()
    )
    assert (
        qualification["source_hashes_unchanged"] and not qualification["changed_hashes"]
    )
    assert qualification["kernel_log_unchanged_entire_batch"]
    report = json.loads((previous / "result.json").read_text())
    assert report["status"] == "failed" and len(report["checks"]) == 6
    assert (
        len(report["steps"]) == 2
        and report["max_accumulation_gradient_relative_l2"] < 2e-6
    )
    root = Path("/tmp/bc250-training-n35-r2")
    checkpoint = root / "checkpoint-step1.pt"
    control = root / "uninterrupted-step2.pt"
    original_sha = qualification["source_sha256"]["validate_training_n35.py"]
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    assert saved["identity"]["test_sha256"] == original_sha
    assert saved["completed_steps"] == 1 and saved["microbatches_seen"] == 16
    assert all(float(s["step"]) == 1 for s in saved["optimizer"]["state"].values())
    del saved
    inputs = {
        p.name: sha(p) for p in (checkpoint, control, root / "independent-fp64-sum.pt")
    }
    output = args.output.parent / "resumed-worker.json"
    command = [
        sys.executable,
        str(Path(__file__).with_name("validate_training_n35.py")),
        "--output",
        str(output),
        "--scratch-root",
        str(root),
        "--resume",
        str(checkpoint),
        "--checkpoint-test-sha",
        original_sha,
        "--state-name",
        "resumed-r3-step2.pt",
    ]
    report.update(
        status="running",
        retained_source_gate=str(previous),
        retained_artifacts_sha256=inputs,
        resume_command=command,
    )
    try:
        with (args.output.parent / "resumed-worker.log").open("w") as log:
            subprocess.run(
                command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=180
            )
        child = json.loads(output.read_text())
        assert child["status"] == "passed"
        actual = torch.load(
            root / child["state_file"], weights_only=True, map_location="cpu"
        )
        expected = torch.load(control, weights_only=True, map_location="cpu")
        assert expected["completed_steps"] == 2 and expected["microbatches_seen"] == 32
        assert all(
            float(s["step"]) == 2 for s in expected["optimizer"]["state"].values()
        )
        identical(actual, expected)
        assert inputs == {
            p.name: sha(p)
            for p in (checkpoint, control, root / "independent-fp64-sum.pt")
        }
        report["checks"] += [
            "fresh-process-exact-372-parameters-gradients-adamw-scheduler",
            "fresh-process-exact-loss-inputs-progress-three-rng",
        ]
        report.update(
            status="passed",
            resumed_worker=child,
            artifacts={
                p.name: dict(bytes=p.stat().st_size, sha256=sha(p))
                for p in sorted(root.glob("*.pt"))
            },
        )
        report.pop("error", None)
    except Exception as error:
        report.update(status="failed", replay_error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
