#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Select stack-owned preservation identities; leave the user's app uninspected."""

import json
from pathlib import Path

from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS

prior = json.loads((WS / "logs/audio-b042-qualification-r6/final.json").read_text())
excluded = [
    n for n in prior["protected_sha256"] if n.startswith("/home/nik/finetune-n7/")
]
for name in excluded:
    del prior["protected_sha256"][name]
for name, expected in prior["protected_sha256"].items():
    assert sha(Path(name)) == expected, name
logs = WS / "logs/codec-b043-entry-r4"
logs.mkdir()
prior.update(
    status="stack-preservation-only-user-application-excluded",
    excluded_user_application_paths=excluded,
    note="User explicitly requested staying out of finetune-n7. No further app reads/edits/tests; no renewed app qualification. Remaining stack identities unchanged.",
)
atomic_json(logs / "final.json", prior)
print(
    "CODEC_B043_STACK_SCOPE",
    len(prior["protected_sha256"]),
    "protected",
    len(excluded),
    "excluded",
)
