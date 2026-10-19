#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Install the requalified trainer and corrected test fixture with rollback."""

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
from qualify_training_n39_on_host import PROJECT, preflight

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--attempt", default="r3")
    args, _ = parser.parse_known_args()
    assert args.attempt == "r3"
    preflight(args.attempt)
    prior = json.loads(
        (WS / "logs/training-n39-qualification-r2/final.json").read_text()
    )
    assert prior["status"] == "passed" and not prior["changed_hashes"]
    for name, expected in prior["protected_sha256"].items():
        assert sha(Path(name)) == expected, name
    for name, expected in prior["source_sha256"].items():
        assert sha(WS / "helpers" / name) == expected, name
    original = json.loads(
        (WS / "logs/training-n39-qualification-r1/final.json").read_text()
    )
    assert original["status"] == "passed"
    for name, expected in original["source_sha256"].items():
        assert sha(WS / "helpers" / name) == expected, name
    unit = json.loads((WS / "logs/training-n39-candidate-unit-tests.json").read_text())
    assert unit["status"] == "passed" and unit["tests"] == 175
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    trainer = PROJECT / "replica/train.py"
    tests = PROJECT / "tests/test_training_launch.py"
    assert (
        sha(tests) == "20115c6efd5c6693ef24e324e98e8b2fb7690d215a929a4a25b0273da1733586"
    )
    scratch = WS / "experiments/training-n39-r1"
    installs = (
        (WS / "helpers/n37_app_train.py", trainer),
        (WS / "helpers/n39_test_training_launch.py", tests),
    )
    assert (
        installs[1][0].read_bytes()
        == (scratch / "tests/test_training_launch.py").read_bytes()
    )
    recovery = WS / "recovery/training-n39-before"
    assert recovery.is_dir()
    before = {}
    for _, target in installs:
        backup = recovery / target.name
        assert not backup.exists()
        shutil.copy2(target, backup)
        before[str(target)] = sha(target)
    staged = copy.deepcopy(prior)
    staged.update(
        status="installed-pending-qualification",
        intended_changes={},
        recovery=str(recovery),
    )
    try:
        for source, target in installs:
            atomic_copy(source, target, target.stat())
            staged["protected_sha256"][str(target)] = sha(target)
            staged["intended_changes"][str(target)] = dict(
                before=before[str(target)], after=sha(target)
            )
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        with (WS / "logs/training-n39-installed-unit-tests.log").open("w") as log:
            subprocess.run(
                ["python3", "-m", "unittest", "discover", "-s", "tests", "-q"],
                cwd=PROJECT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        directory = WS / "logs/training-n39-deployment-r3"
        directory.mkdir()
        atomic_json(directory / "final.json", staged)
        main(
            batch_id="training-n39",
            baseline_label="training-n39-deployment-r3",
            default_phase="extra",
            extra_gates=(
                (
                    "installed",
                    "validate_installed_training_n37.py",
                    ["--scratch-root", str(scratch)],
                    4,
                ),
            ),
            additional_sources=(
                "deploy_training_n39_on_host.py",
                "qualify_training_n39_on_host.py",
                "validate_installed_training_n37.py",
                "n37_app_train.py",
                "n39_test_training_launch.py",
                "bc250_checkpoint_provenance.py",
            ),
            scratch_mounts=(scratch, Path("/tmp/bc250-training-n36-r1")),
        )
    except Exception:
        assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
        for _, target in installs:
            shutil.copy2(target, recovery / (target.name + ".failed-new"))
            atomic_copy(recovery / target.name, target, target.stat())
        atomic_json(
            WS / "logs/training-n39-rollback.json",
            dict(status="restored-original-application", files=before),
        )
        raise
