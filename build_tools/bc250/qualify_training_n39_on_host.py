#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Reconcile trainer features in an owned overlay before guarded installation."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil

from deploy_torch_a04_on_host import sha
from qualify_training_n33_on_host import WS, main

PROJECT = Path("/home/nik/finetune-n7")


def preflight(attempt):
    assert os.geteuid() == 0
    available = (
        int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
        )
        * 1024
    )
    assert available > 8 * 1024**3, available
    clients = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            links = [str(p.readlink()) for p in (proc / "fd").iterdir()]
            if any("/dev/kfd" in p or "/dev/dri/render" in p for p in links):
                comm, uid = (proc / "comm").read_text().strip(), proc.stat().st_uid
                clients.append(dict(pid=int(proc.name), comm=comm, uid=uid))
        except OSError:
            continue
    assert all(
        p["uid"] == 0 and p["comm"].startswith("cyan-skillfish") for p in clients
    ), clients
    (WS / "logs" / ("training-n39-resource-" + attempt + ".json")).write_text(
        json.dumps(
            dict(memory_available_bytes=available, gpu_clients=clients), indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--attempt", default="r1")
    args, _ = parser.parse_known_args()
    assert re.fullmatch(r"r[1-9][0-9]*", args.attempt)
    preflight(args.attempt)
    baseline = json.loads(
        (WS / "logs/training-n38-qualification-r9/final.json").read_text()
    )
    assert baseline["status"] == "passed"
    for name, expected in baseline["protected_sha256"].items():
        assert sha(Path(name)) == expected, name
    assert (
        sha(WS / "helpers/n37_app_train.py")
        == "b9bf2587d8f3b14b3f7ca7f62bcfface0c1a72606633ac5920600c28f1f0f4c1"
    )
    assert (PROJECT / "replica/bc250_checkpoint_provenance.py").read_bytes() == (
        WS / "helpers/bc250_checkpoint_provenance.py"
    ).read_bytes()
    assert os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize > 4 * 1024**3
    scratch = WS / "experiments" / ("training-n39-" + args.attempt)
    scratch.mkdir(parents=True)
    app = scratch / "application/replica"
    app.mkdir(parents=True)
    for p in (PROJECT / "replica").glob("*.py"):
        if p.name == "train.py":
            shutil.copy2(WS / "helpers/n37_app_train.py", app / p.name)
        else:
            (app / p.name).symlink_to(p)
    # Keep the original trainer as read-only evidence for fingerprint comparison.
    shutil.copy2(PROJECT / "replica/train.py", scratch / "entry_train.py")
    runtime = json.loads(Path("/opt/bc250-stack/current/runtime.json").read_text())
    for p in (scratch, *scratch.rglob("*")):
        if not p.is_symlink():
            os.chown(p, runtime["uid"], runtime["gid"])
    main(
        batch_id="training-n39",
        baseline_label="training-n38-qualification-r9",
        default_phase="extra",
        extra_gates=tuple(
            (
                mode,
                "validate_training_n39.py",
                ["--scratch-root", str(scratch), "--mode", mode],
                count,
            )
            for mode, count in (("fresh", 4), ("resumed", 5), ("legacy", 5))
        )
        + (
            (
                "refusals",
                "validate_provenance_n37.py",
                ["--scratch-root", str(scratch)],
                8,
            ),
            (
                "attention",
                "validate_attention_cli_n39.py",
                ["--scratch-root", str(scratch)],
                4,
            ),
        ),
        additional_sources=(
            "validate_training_n39.py",
            "validate_training_n37.py",
            "validate_qwen35_training_cli.py",
            "n37_app_train.py",
            "bc250_checkpoint_provenance.py",
            "validate_provenance_n37.py",
            "validate_attention_cli_n39.py",
        ),
        scratch_mounts=(scratch, Path("/tmp/bc250-training-n36-r1")),
    )
