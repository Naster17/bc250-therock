#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify actual owned N3.7 checkpoint refusal for the preserved cosmetic edit."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from validate_qwen35_training_cli import PROJECT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.scratch_root
    old = Path("/tmp/bc250-training-n37-r1/fresh/checkpoint-00000001")
    fresh = json.loads((root / "fresh-result.json").read_text())
    record = json.loads((old / "runtime-provenance.json").read_text())
    current = json.loads(
        (root / "fresh/checkpoint-00000001/runtime-provenance.json").read_text()
    )
    assert record["fingerprint"] == current["fingerprint"]
    before = {
        p.name: hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
        for p in old.iterdir()
        if p.is_file()
    }
    differences = {
        k
        for k in record["identity"]["files"]
        if record["identity"]["files"][k] != current["identity"]["files"][k]
    }
    assert differences == {"application/bc250_ops.py"}, differences
    old_identity = dict(record["identity"])
    current_identity = dict(current["identity"])
    old_identity.pop("files")
    current_identity.pop("files")
    assert old_identity == current_identity
    env = dict(os.environ)
    env.update(
        PYTHONPATH=f"{PROJECT}/runtime/transformers-5.19.0-cp312:{root}/application:"
        + env.get("PYTHONPATH", ""),
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_XET="1",
    )
    output = root / "refused-n37-original-ops"
    command = fresh["command"][:]
    command[command.index("--output") + 1] = str(output)
    command += ["--resume", str(old)]
    logpath = root / "n37-original-ops-refusal.log"
    with logpath.open("w") as log:
        result = subprocess.run(
            command,
            env=env,
            cwd=root / "application",
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=90,
        )
    log = logpath.read_text()
    assert (
        result.returncode == 2 and "Checkpoint numerical runtime mismatch" in log
    ), log[-3000:]
    assert not output.exists() and "Loading local text-only base weights" not in log
    after = {
        p.name: hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
        for p in old.iterdir()
        if p.is_file()
    }
    assert before == after
    args.output.write_text(
        json.dumps(
            dict(
                status="passed",
                checks=[
                    "owned-n37-cosmetic-hook-change-refused-before-weights-no-mutation"
                ],
                differing_identity_files=sorted(differences),
                checkpoint_sha256=before,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
