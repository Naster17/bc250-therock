#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Capture a native stack; a reproduced fault is not numerical qualification."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--fixture",
    choices=("minimal", "full", "unprofiled", "instrumented"),
    default="full",
)
args = parser.parse_args()
log = args.output.parent / "gdb.log"
full = Path(__file__).with_name("validate_audio_b042.py")
if args.fixture == "unprofiled":
    original = full.read_text()
    assert original.count("trace=True,") == 1
    full = args.output.parent / "diagnostic-unprofiled.py"
    full.write_text(original.replace("trace=True,", "trace=False,"))
if args.fixture == "instrumented":
    original = full.read_text()
    full = args.output.parent / "diagnostic-instrumented.py"
    assert original.count("with torch.cuda.stream(current):") == 1
    full.write_text(
        original.replace(
            "with torch.cuda.stream(current):",
            'print("CASE", name, "STREAM", current.cuda_stream, flush=True)\n        with torch.cuda.stream(current):',
        )
    )
commands = args.output.parent / "gdb-commands.txt"
commands.write_text(
    'set breakpoint pending on\nbreak hipLaunchKernel\ncommands\nsilent\nprintf "HIP_LAUNCH func=%p stream=%p\\n", $rdi, *(void**)($rsp+16)\ncontinue\nend\n'
)
with log.open("w") as output:
    result = subprocess.run(
        [
            "gdb",
            "--batch",
            "--return-child-result",
            *(["-x", str(commands)] if args.fixture == "instrumented" else []),
            "-ex",
            "set pagination off",
            "-ex",
            "run",
            "-ex",
            "thread apply all bt 8",
            "-ex",
            "bt 30",
            "--args",
            sys.executable,
            *(
                [str(Path(__file__).with_name("reproduce_audio_rnnt_b042.py"))]
                if args.fixture == "minimal"
                else [
                    str(full),
                    "--output",
                    str(args.output.parent / "numerical-attempt.json"),
                    "--group",
                    "native",
                ]
            ),
        ],
        stdout=output,
        stderr=subprocess.STDOUT,
        timeout=180,
    )
args.output.write_text(
    json.dumps(
        dict(
            status="diagnostic-only",
            gdb_returncode=result.returncode,
            checks=[],
            log=str(log),
        ),
        indent=2,
    )
    + "\n"
)
