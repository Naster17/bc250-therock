#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Use explicit RAM-backed scratch for actual CLI resume on a nearly full disk."""

import argparse
import json
import os
from pathlib import Path
import re

from qualify_training_n33_on_host import WS, main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--attempt", default="r1")
    args, _ = parser.parse_known_args()
    assert re.fullmatch(r"r[1-9][0-9]*", args.attempt)
    assert os.geteuid() == 0
    assert any(
        line.split()[1:3] == ["/tmp", "tmpfs"]
        for line in Path("/proc/mounts").read_text().splitlines()
    )
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
    temporary_free = os.statvfs("/tmp").f_bavail * os.statvfs("/tmp").f_frsize
    assert available > 6 * 1024**3 and temporary_free > 2 * 1024**3
    scratch = Path("/tmp") / ("bc250-training-n36-" + args.attempt)
    scratch.mkdir(mode=0o700)
    manifest = json.loads(Path("/opt/bc250-stack/current/runtime.json").read_text())
    os.chown(scratch, manifest["uid"], manifest["gid"])
    (WS / "logs" / ("training-n36-preflight-" + args.attempt + ".json")).write_text(
        json.dumps(
            dict(
                scratch=str(scratch),
                scratch_filesystem="tmpfs",
                memory_available_bytes=available,
                scratch_available_bytes=temporary_free,
                minimum_disk_free_bytes=512 * 1024**2,
                purpose="Only text logs on disk; preserve checkpoints in RAM then copy/hash-verify on local workspace before handoff",
            ),
            indent=2,
        )
        + "\n"
    )
    main(
        batch_id="training-n36",
        baseline_label="training-n35-qualification-r3",
        default_phase="extra",
        extra_gates=tuple(
            (
                mode,
                "validate_training_n36.py",
                ["--scratch-root", str(scratch), "--mode", mode],
                count,
            )
            for mode, count in (("fresh", 3), ("resumed", 4))
        ),
        additional_sources=(
            "validate_training_n36.py",
            "validate_qwen35_training_cli.py",
        ),
        scratch_mounts=(scratch,),
        minimum_disk_free_bytes=512 * 1024**2,
    )
