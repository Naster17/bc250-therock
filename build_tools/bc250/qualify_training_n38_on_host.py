#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run vocabulary layout/reference gates separately on the preserved native stack."""

import argparse
import json
from pathlib import Path
import sys

from qualify_training_n33_on_host import WS, main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--stage",
        choices=(
            "layout",
            "forward",
            "input-backward",
            "policy-forward",
            "policy-backward",
            "model",
            "model-stable",
        ),
        required=True,
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    if args.stage == "model-stable":
        memory = {
            line.split(":", 1)[0]: int(line.split()[1]) * 1024
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith(("MemAvailable:", "SwapFree:"))
        }
        clients = []
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                links = [str(p.readlink()) for p in (proc / "fd").iterdir()]
                if any(
                    "/dev/kfd" in link or "/dev/dri/render" in link for link in links
                ):
                    clients.append(
                        dict(
                            pid=int(proc.name),
                            comm=(proc / "comm").read_text().strip(),
                            uid=proc.stat().st_uid,
                        )
                    )
            except (OSError, PermissionError):
                continue
        occupied = [
            p
            for p in clients
            if not (p["uid"] == 0 and p["comm"].startswith("cyan-skillfish"))
        ]
        assert not occupied, f"GPU already held by other processes: {occupied}"
        assert memory["MemAvailable"] > 8 * 1024**3, memory
        attempt = remaining[remaining.index("--attempt") + 1]
        (WS / "logs" / ("training-n38-resource-" + attempt + ".json")).write_text(
            json.dumps(
                dict(
                    memory=memory,
                    clients=clients,
                    note="Exclusive prefix lock does not serialize raw GPU clients; keep other work idle during this gate.",
                ),
                indent=2,
            )
            + "\n"
        )
    if args.stage == "layout":
        gates = (("layout", "audit_vocab_n38.py", [], 2),)
    elif args.stage in ("model", "model-stable"):
        gates = (
            (
                args.stage,
                "validate_model_vocab_n38.py",
                ["--settle-steps", "2"] if args.stage == "model-stable" else [],
                4,
            ),
        )
    else:
        role = (
            "forward"
            if args.stage in ("forward", "policy-forward")
            else "input-backward"
        )
        arguments = ["--role", role]
        if args.stage.startswith("policy-"):
            arguments.append("--match-torch")
        gates = ((args.stage, "benchmark_vocab_n38.py", arguments, 1),)
    main(
        batch_id="training-n38",
        baseline_label="training-n38-entry-r2",
        default_phase="extra",
        extra_gates=gates,
        additional_sources=(
            "audit_vocab_n38.py",
            "benchmark_vocab_n38.py",
            "n38_qualified_bc250_ops.py",
            *(
                (
                    "validate_model_vocab_n38.py",
                    "validate_model_gemm_n32.py",
                    "rocblas_gemm_n38.csv",
                )
                if args.stage in ("model", "model-stable")
                else ()
            ),
        ),
        minimum_disk_free_bytes=6 * 1024**3 if args.stage == "layout" else 2 * 1024**3,
    )
