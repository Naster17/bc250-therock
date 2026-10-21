#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Relink retained HIP objects into an isolated symbol file; no runtime writes."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import hashlib

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--directory", type=Path, default=Path("/workspace/audio-b042/hip-symbols-r2")
)
parser.add_argument(
    "--reference",
    type=Path,
    default=Path("/workspace/build/core/clr/build/hipamd/lib/libamdhip64.so"),
)
args_cli = parser.parse_args()
root = args_cli.directory
root.mkdir()
build = Path("/workspace/build/core/clr/build")
commands = subprocess.check_output(
    [
        "ninja",
        "-C",
        str(build),
        "-t",
        "commands",
        "hipamd/lib/libamdhip64.so.7.17.0-0000000",
    ],
    text=True,
)
(root / "original-link-commands.txt").write_text(commands)
clauses = [c.strip() for c in commands.splitlines()[-1].split("&&")]
clause = next(c for c in clauses if "-shared" in c)
args = shlex.split(clause)
args = [a for a in args if a not in ("-s", "-Wl,--strip-all", "-Wl,-s")]
args[args.index("-o") + 1] = str(root / "libamdhip64-symbols.so")
for i, arg in enumerate(args):
    if arg.startswith("--dependency-file=") or arg.startswith("-Wl,--dependency-file="):
        args[i] = arg.split("=")[0] + "=" + str(root / "link.d")
(root / "link-command.json").write_text(json.dumps(args, indent=2) + "\n")
subprocess.run(args, cwd=build, check=True, timeout=180)
# A symbol-only interpretation requires identical machine code.
objects = {}
for label, library in (
    ("qualified", args_cli.reference),
    ("symbols", root / "libamdhip64-symbols.so"),
):
    text = root / (label + ".text")
    subprocess.run(
        [
            "/workspace/build/dist/rocm/lib/llvm/bin/llvm-objcopy",
            "--dump-section",
            ".text=" + str(text),
            str(library),
            str(root / (label + "-inspection-copy.so")),
        ],
        check=True,
    )
    objects[label] = hashlib.file_digest(text.open("rb"), "sha256").hexdigest()
(root / "verification.json").write_text(json.dumps(objects, indent=2) + "\n")
print(objects, flush=True)
assert objects["qualified"] == objects["symbols"]
