#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Preserve two reviewed documentation drifts while pinning all native identities."""

import json
import os
from pathlib import Path

from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS

if __name__ == "__main__":
    assert os.geteuid() == 0
    prior = json.loads(
        (WS / "logs/training-n310-qualification-r1/final.json").read_text()
    )
    assert prior["status"] == "passed"
    admitted = {
        "/home/nik/finetune-n7/BC250_TRAINING_OPS.md": "084843ea94eaf70233c05045c36858a7d089b1435dd2ef4c1f9ccac7449bca2c",
        "/home/nik/finetune-n7/AGENTS.md": "8de76ba12a47634e1ec15802f696e683e2f66c059435d55254f74bc35298d1a1",
    }
    changes = {}
    for name, expected in prior["protected_sha256"].items():
        actual = sha(Path(name))
        if name in admitted:
            assert actual == admitted[name] and actual != expected, name
            changes[name] = dict(qualified=expected, entry=actual)
        else:
            assert actual == expected, name
        prior["protected_sha256"][name] = actual
    prior.update(
        status="recorded-documentation-entry-drift",
        entry_changes=changes,
        note="Documentation only; native/source/app identities otherwise preserved. This is a preservation snapshot, not renewed trainer/documentation qualification. No application files changed.",
    )
    destination = WS / "logs/vision-b041-entry-r2"
    destination.mkdir()
    atomic_json(destination / "final.json", prior)
