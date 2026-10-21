#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Preserve one reviewed user app-document change without app qualification."""

import json
from pathlib import Path

from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS

prior = json.loads((WS / "logs/audio-b042-qualification-r6/final.json").read_text())
admitted = {
    "/home/nik/finetune-n7/AGENTS.md": "7cefad6be76f8ff9c36858fa622b14b5b1ea35ad066a3d78df08efbcd1c56a36"
}
changes = {}
for name, expected in prior["protected_sha256"].items():
    actual = sha(Path(name))
    if actual != expected:
        assert admitted.get(name) == actual, name
        changes[name] = dict(previous=expected, entry=actual)
        prior["protected_sha256"][name] = actual
assert list(changes) == list(admitted)
logs = WS / "logs/codec-b043-entry-r2"
logs.mkdir()
prior.update(
    status="preservation-only-reviewed-document-drift",
    entry_document_drift=changes,
    note="User Burn FP16/recomputation evidence documentation preserved; not renewed trainer qualification.",
)
atomic_json(logs / "final.json", prior)
