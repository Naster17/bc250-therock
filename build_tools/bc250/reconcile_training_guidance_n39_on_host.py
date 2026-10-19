#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Synchronize only stale attention guidance after installed GPU qualification."""

import json
import os
from pathlib import Path
import shutil
import subprocess

from deploy_qwen35_training_ops import atomic_copy
from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS

if __name__ == "__main__":
    assert os.geteuid() == 0
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    prior = json.loads(
        (WS / "logs/training-n39-qualification-r3/final.json").read_text()
    )
    assert prior["status"] == "passed" and not prior["changed_hashes"]
    for name, expected in prior["protected_sha256"].items():
        assert sha(Path(name)) == expected, name
    target = Path("/home/nik/finetune-n7/AGENTS.md")
    assert (
        sha(target)
        == "6c17635ba8cb5cc7686e2413005792fef4939beb9876760527069483813f46a8"
    )
    original = target.read_text()
    old, new = (
        "  - Math SDPA only: `sdpa_kernel(SDPBackend.MATH)` must wrap all forward and backward passes.\n  - FlashAttention is not supported on gfx1013. Tiled attention is experimental and unqualified for Qwen.\n",
        "  - Math SDPA remains the default for forward/backward and legacy math fingerprints.\n  - Explicit `--attention-backend tiled` is bounded-qualified by the BC-250 workspace A07/N3.9 evidence. Head256 BF16 uses math forward/tiled backward; this is not FlashAttention or a universal speedup.\n  - Keep backend and runtime identities when resuming. New provenance-bound checkpoints reject even cosmetic hook-source changes; never silently remove or migrate their provenance. See `BC250_TRAINING_OPS.md` and the canonical workspace `flow.md`.\n",
    )
    assert original.count(old) == 1
    recovery = WS / "recovery/training-n39-before"
    assert not (recovery / target.name).exists()
    shutil.copy2(target, recovery / target.name)
    candidate = recovery / "AGENTS.n39-candidate.md"
    candidate.write_text(original.replace(old, new))
    before = subprocess.check_output(["dmesg"])
    atomic_copy(candidate, target, target.stat())
    prior["protected_sha256"][str(target)] = sha(target)
    for name, expected in prior["protected_sha256"].items():
        assert sha(Path(name)) == expected, name
    assert before == subprocess.check_output(["dmesg"])
    directory = WS / "logs/training-n39-guidance-r4"
    directory.mkdir()
    prior.update(
        guidance_only=True,
        gpu_qualification="training-n39-qualification-r3",
        guidance_change=dict(
            path=str(target),
            before="6c17635ba8cb5cc7686e2413005792fef4939beb9876760527069483813f46a8",
            after=sha(target),
        ),
        documentation_source_sha256=sha(Path(__file__)),
    )
    atomic_json(directory / "final.json", prior)
