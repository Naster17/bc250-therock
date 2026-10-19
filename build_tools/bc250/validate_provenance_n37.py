#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Refuse owned corrupted/changed-runtime checkpoints before model weights load."""

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.scratch_root
    sys.path.insert(0, str(root / "application"))
    from replica import bc250_checkpoint_provenance as provenance

    fresh = json.loads((root / "fresh-result.json").read_text())
    checkpoint = root / "fresh/checkpoint-00000001"
    original_run = json.loads((checkpoint / "run.json").read_text())
    original = json.loads((checkpoint / provenance.FILENAME).read_text())
    report = dict(status="running", checks=[])
    try:
        for mode in (
            "missing",
            "malformed",
            "digest",
            "schema",
            "orphan",
            "preset",
            "hook",
            "native",
        ):
            fixture = root / ("refusal-" + mode)
            fixture.mkdir()
            for file in checkpoint.iterdir():
                if file.name not in ("run.json", provenance.FILENAME):
                    (fixture / file.name).symlink_to(file)
            run, record = copy.deepcopy(original_run), copy.deepcopy(original)
            if mode == "schema":
                record["schema"] = 2
                run[provenance.MARKER] = provenance.digest(record)
            if mode == "native":
                record["identity"]["files"]["torch/_C"] = "0" * 64
                run[provenance.MARKER] = provenance.digest(record)
            if mode == "orphan":
                del run[provenance.MARKER]
            (fixture / "run.json").write_text(json.dumps(run))
            if mode != "missing":
                (fixture / provenance.FILENAME).write_text(
                    "{" if mode == "malformed" else json.dumps(record)
                )
            if mode == "digest":
                (fixture / provenance.FILENAME).write_text(
                    json.dumps({**record, "fingerprint": "0" * 64})
                )
            env = dict(os.environ)
            env.update(
                PYTHONPATH=str(root / "application")
                + ":/home/nik/finetune-n7/runtime/transformers-5.19.0-cp312:"
                + env.get("PYTHONPATH", ""),
                OMP_NUM_THREADS="4",
                MKL_NUM_THREADS="4",
                HF_HUB_OFFLINE="1",
                HF_HUB_DISABLE_XET="1",
            )
            cwd = root / "application"
            if mode == "preset":
                preset = root / "changed-preset.csv"
                preset.write_bytes(
                    Path(env["ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH"]).read_bytes() + b"\n"
                )
                env["ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH"] = str(preset)
            if mode == "hook":
                cwd = root / "changed-hook"
                package = cwd / "replica"
                package.mkdir(parents=True)
                for path in (root / "application/replica").glob("*.py"):
                    if path.name == "bc250_ops.py":
                        (package / path.name).write_bytes(
                            path.read_bytes()
                            + b"\n# Owned provenance refusal fixture.\n"
                        )
                    else:
                        (package / path.name).symlink_to(path)
            output = root / ("refused-output-" + mode)
            command = fresh["command"][:]
            command[command.index("--output") + 1] = str(output)
            command += ["--resume", str(fixture)]
            if mode in ("missing", "malformed", "digest", "schema", "orphan"):
                command += ["--preflight"]
            with (root / ("refusal-" + mode + ".log")).open("w") as log:
                result = subprocess.run(
                    command,
                    cwd=cwd,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=90,
                )
            log = (root / ("refusal-" + mode + ".log")).read_text()
            assert (
                result.returncode == 2 and "provenance" in log.lower()
                if mode not in ("preset", "hook", "native")
                else result.returncode == 2
                and "Checkpoint numerical runtime mismatch" in log
            ), log[-4000:]
            assert (
                "Loading local text-only base weights" not in log
                and not output.exists()
            )
            report["checks"].append(
                dict(name=mode, exit_code=2, before_weights=True, no_output=True)
            )
        report["status"] = "passed"
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
