#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Retain passed training gates; rerun refusals after recorded network-only log delta."""

import json
import os
from pathlib import Path

from deploy_torch_a04_on_host import atomic_json, check_gate, sha
from qualify_training_n33_on_host import WS, main

if __name__ == "__main__":
    prior_dir = WS / "logs/training-n37-qualification-r1"
    prior = json.loads((prior_dir / "final.json").read_text())
    assert (
        prior["status"] == "failed"
        and prior["error"] == "AssertionError('training-n37-r1-refusals')"
    )
    assert not prior["changed_hashes"] and prior["source_hashes_unchanged"]
    for path, expected in prior["protected_sha256"].items():
        assert sha(Path(path)) == expected, path
    for name, expected in prior["source_sha256"].items():
        assert sha(WS / "helpers" / name) == expected, name
    retained = []
    for mode in ("fresh", "resumed", "legacy"):
        label = "training-n37-r1-" + mode
        check_gate(label)
        retained.append(
            dict(label=label, result_sha256=sha(WS / "logs" / label / "result.json"))
        )
    failed = WS / "logs/training-n37-r1-refusals"
    before = (failed / "kernel-before.log").read_bytes()
    after = (failed / "kernel-after.log").read_bytes()
    assert after.startswith(before)
    additions = after[len(before) :].decode().splitlines()
    assert len(additions) == 2
    assert additions[0].endswith("Initializing XFRM netlink socket")
    assert additions[1].endswith(
        "bridge: filtering via arp/ip/ip6tables is no longer available by default. Update your scripts to load br_netfilter if you need this."
    )
    scratch = Path("/tmp/bc250-training-n37-refusals-r2")
    scratch.mkdir(mode=0o700)
    original = Path("/tmp/bc250-training-n37-r1")
    for name in ("application", "fresh", "fresh-result.json"):
        (scratch / name).symlink_to(original / name)
    manifest = json.loads(Path("/opt/bc250-stack/current/runtime.json").read_text())
    os.chown(scratch, manifest["uid"], manifest["gid"])
    main(
        batch_id="training-n37",
        baseline_label="training-n37-qualification-r1",
        default_phase="extra",
        extra_gates=(
            (
                "refusals",
                "validate_provenance_n37.py",
                ["--scratch-root", str(scratch)],
                8,
            ),
        ),
        additional_sources=(
            "validate_provenance_n37.py",
            "n37_app_train.py",
            "bc250_checkpoint_provenance.py",
            "validate_training_n37.py",
        ),
        scratch_mounts=(scratch, original),
        minimum_disk_free_bytes=512 * 1024**2,
    )
    final_path = WS / "logs/training-n37-qualification-r2/final.json"
    final = json.loads(final_path.read_text())
    final.update(
        retained_training_gates=retained,
        original_failed_gate="training-n37-r1-refusals",
        original_kernel_additions=additions,
        original_training_source_sha256=prior["source_sha256"],
        acceptance="Original training gates remain unchanged/passed; corrected refusal gate requires byte-identical dmesg. Original full batch remains failed.",
    )
    atomic_json(final_path, final)
