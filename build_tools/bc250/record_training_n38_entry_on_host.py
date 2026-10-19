#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Record two audited application drifts; reject any other protected identity change."""

import json
import os
from pathlib import Path
from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS

if __name__ == "__main__":
    assert os.geteuid() == 0
    prior = json.loads(
        (WS / "logs/training-n37-qualification-r3/final.json").read_text()
    )
    admitted = {
        "/home/nik/finetune-n7/replica/train.py": "170d39204518bcd9526f86dba3f67c40cfeafbd547f764796e366dbb9e02ac8e",
        "/home/nik/finetune-n7/replica/bc250_ops.py": "d058b14a4bae969e5b84a9de8f69706c0f6e3df910409dfc15843f0eeee3d41a",
    }
    changes = {}
    for name, expected in prior["protected_sha256"].items():
        actual = sha(Path(name))
        if name in admitted:
            assert actual == admitted[name] and actual != expected, name
            changes[name] = dict(before=expected, entry=actual)
        else:
            assert actual == expected, name
        prior["protected_sha256"][name] = actual
    assert (
        sha(WS / "helpers/n38_qualified_bc250_ops.py")
        == "f0bcb35eadd3cde6c08dbeca0929358e4361f1f2b7358f2cad5288517f9fa4e6"
    )
    destination = WS / "logs/training-n38-entry-r2"
    destination.mkdir()
    prior.update(
        status="recorded-entry-drift-not-application-qualification",
        entry_changes=changes,
        isolated_hook_sha256=sha(WS / "helpers/n38_qualified_bc250_ops.py"),
        note="Existing app is preserved. Trainer lacks N37/tiled CLI; hook differs only in printed status. N38 isolated tests use unchanged qualified hook bytes; no production app or checkpoint rewrite.",
    )
    atomic_json(destination / "final.json", prior)
