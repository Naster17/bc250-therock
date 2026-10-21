#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compile one CLR TU and relink private copies of its archive and HIP DSO."""

import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

root = Path("/workspace/audio-b042/runtime-candidate-r1")
root.mkdir()
build = Path("/workspace/build/core/clr/build")
commands = (
    (Path("/workspace/audio-b042/hip-symbols/original-link-commands.txt"))
    .read_text()
    .splitlines()
)
source = "/workspace/source/rocm-systems/projects/clr/rocclr/device/rocm/rocvirtual.cpp"
compile = shlex.split(next(c for c in commands if " -c " + source in c))
for flag, value in (
    ("-o", str(root / "rocvirtual.cpp.o")),
    ("-MF", str(root / "rocvirtual.cpp.o.d")),
    ("-MT", str(root / "rocvirtual.cpp.o")),
):
    compile[compile.index(flag) + 1] = value
(root / "compile-command.json").write_text(json.dumps(compile, indent=2) + "\n")
subprocess.run(compile, cwd=build, check=True, timeout=600)
archive = root / "librocclr.a"
shutil.copy2(build / "rocclr/librocclr.a", archive)
ar = "/workspace/build/dist/rocm/lib/llvm/bin/llvm-ar"
assert (
    subprocess.check_output([ar, "t", str(archive)], text=True)
    .splitlines()
    .count("rocvirtual.cpp.o")
    == 1
)
subprocess.run([ar, "r", str(archive), str(root / "rocvirtual.cpp.o")], check=True)
link = shlex.split(next(c.strip() for c in commands[-1].split("&&") if "-shared" in c))
link = [a for a in link if a not in ("-s", "-Wl,--strip-all", "-Wl,-s")]
link[link.index("-o") + 1] = str(root / "libamdhip64.so.7.17.0-0000000")
assert link.count("rocclr/librocclr.a") == 1
link[link.index("rocclr/librocclr.a")] = str(archive)
for i, arg in enumerate(link):
    if arg.startswith("--dependency-file=") or arg.startswith("-Wl,--dependency-file="):
        link[i] = arg.split("=")[0] + "=" + str(root / "link.d")
(root / "link-command.json").write_text(json.dumps(link, indent=2) + "\n")
subprocess.run(link, cwd=build, check=True, timeout=180)
report = dict(
    status="passed",
    source_sha256=hashlib.file_digest(Path(source).open("rb"), "sha256").hexdigest(),
    candidate_sha256=hashlib.file_digest(
        (root / "libamdhip64.so.7.17.0-0000000").open("rb"), "sha256"
    ).hexdigest(),
    original_sha256=hashlib.file_digest(
        Path("/workspace/build/dist/rocm/lib/libamdhip64.so").open("rb"), "sha256"
    ).hexdigest(),
)
(root / "final.json").write_text(json.dumps(report, indent=2) + "\n")
print(report, flush=True)
