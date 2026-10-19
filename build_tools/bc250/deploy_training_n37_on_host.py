#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Deploy candidate-tested checkpoint provenance under the exclusive runtime lock."""

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess

from deploy_qwen35_training_ops import atomic_copy
from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS, main

PROJECT = Path("/home/nik/finetune-n7")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--attempt", default="r2")
    args, _ = parser.parse_known_args()
    assert args.attempt == "r2" and os.geteuid() == 0
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    prior = json.loads(
        (WS / "logs/training-n37-qualification-r1/final.json").read_text()
    )
    assert prior["status"] == "passed" and not prior["changed_hashes"]
    for path, expected in prior["protected_sha256"].items():
        assert sha(Path(path)) == expected, path
    trainer = PROJECT / "replica/train.py"
    module = PROJECT / "replica/bc250_checkpoint_provenance.py"
    assert not module.exists()
    recovery = WS / "recovery/training-n37-before-r2"
    recovery.mkdir()
    doc = PROJECT / "BC250_TRAINING_OPS.md"
    for path in (trainer, doc):
        shutil.copy2(path, recovery / path.name)
    installs = (
        ("n37_app_train.py", trainer),
        (module.name, module),
        ("qwen35_training_usage.md", doc),
    )
    before_hashes = {
        str(target): sha(target) if target.exists() else None for _, target in installs
    }
    staged = copy.deepcopy(prior)
    staged.update(
        status="installed-pending-gpu-qualification",
        recovery=str(recovery),
        intended_changes={},
    )
    try:
        for source, target in installs:
            atomic_copy(WS / "helpers" / source, target, trainer.stat())
            staged["protected_sha256"][str(target)] = sha(target)
            staged["intended_changes"][str(target)] = dict(
                before=before_hashes[str(target)], after=sha(target)
            )
        directory = WS / "logs/training-n37-deployment-r2"
        directory.mkdir()
        atomic_json(directory / "final.json", staged)
        main(
            batch_id="training-n37",
            baseline_label="training-n37-deployment-r2",
            default_phase="extra",
            extra_gates=(
                (
                    "installed",
                    "validate_installed_training_n37.py",
                    ["--scratch-root", "/tmp/bc250-training-n37-r1"],
                    4,
                ),
            ),
            additional_sources=(
                "validate_installed_training_n37.py",
                "n37_app_train.py",
                module.name,
            ),
            scratch_mounts=(
                Path("/tmp/bc250-training-n37-r1"),
                Path("/tmp/bc250-training-n36-r1"),
            ),
        )
    except Exception:
        # Keep the failed candidate and restore the old app; never delete user data.
        assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
        for path in (trainer, doc):
            shutil.copy2(path, recovery / (path.name + ".failed-new"))
            atomic_copy(recovery / path.name, path, path.stat())
        if module.exists():
            module.rename(recovery / module.name)
        raise
