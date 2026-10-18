#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build an independent Torch candidate for the bounded sparse-pivot correction."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

WS = Path("/home/nik/bc250work/rocm-10.2-gfx1013")
CANDIDATE = WS / "pytorch-sparse-n213"


def main() -> None:
    assert os.geteuid() == 0
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    source = (Path("/opt/bc250-stack/current") / "torch").resolve()
    assert source == WS / "pytorch-profiler-order"
    assert (
        source / "GIT_REVISION"
    ).read_text().strip() == "b823a1b2f327d5ba07af0796876f93e5308a1012"
    assert os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize > 7 * 1024**3
    assert not CANDIDATE.exists()
    CANDIDATE.mkdir()
    # Independent copies: no writable source/object/DSO hardlinks to the active
    # build. Tar retains only hardlinks internal to the new copy.
    writer = subprocess.Popen(
        ["tar", "-C", str(source), "--exclude=./third_party", "-cf", "-", "."],
        stdout=subprocess.PIPE,
    )
    subprocess.run(
        ["tar", "-C", str(CANDIDATE), "-xf", "-"], stdin=writer.stdout, check=True
    )
    writer.stdout.close()
    assert writer.wait() == 0
    (CANDIDATE / "third_party").mkdir(exist_ok=True)
    staged = WS / "staging/sparse-n213"
    for filename, target in [
        ("SparseBlasImpl.h", "aten/src/ATen/native/sparse/cuda/SparseBlasImpl.h"),
        ("SparseBlas.cpp", "aten/src/ATen/native/sparse/cuda/SparseBlas.cpp"),
        (
            "SparseCsrTensorMath.cu",
            "aten/src/ATen/native/sparse/cuda/SparseCsrTensorMath.cu",
        ),
        ("test_bc250_sparse_triangular.py", "test/test_bc250_sparse_triangular.py"),
    ]:
        (CANDIDATE / target).write_bytes((staged / filename).read_bytes())
    (CANDIDATE / "GIT_REVISION").write_bytes((staged / "GIT_REVISION").read_bytes())
    logs = WS / "logs/sparse-n213-build"
    logs.mkdir()
    command = [
        "podman",
        "run",
        "--name",
        "bc250-sparse-n213-build",
        "--network",
        "none",
        "-e",
        "BC250_A09_FRESH=0",
    ]
    for src, dst, mode in [
        (WS / "source", "/workspace/source", "ro"),
        (WS / "helpers", "/workspace/helpers", "ro"),
        (CANDIDATE, "/workspace/pytorch-work", "rw"),
        (
            WS / "pytorch-sparse-followup/third_party",
            "/workspace/pytorch-work/third_party",
            "ro",
        ),
        (WS / "build", "/workspace/build", "ro"),
        (WS / "cache", "/workspace/cache", "rw"),
        (logs, "/workspace/logs", "rw"),
    ]:
        command += ["-v", f"{src}:{dst}:{mode}"]
    command += [
        "localhost/bc250-pytorch-build:2.13-gfx1013",
        "bash",
        "/workspace/helpers/build_torch_profiler_order.sh",
    ]
    (logs / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    before = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            source / "aten/src/ATen/native/sparse/cuda/SparseCsrTensorMath.cu",
            source / "torch/lib/libtorch_cpu.so",
            source / "GIT_REVISION",
        ]
    }
    (logs / "protected-active.json").write_text(json.dumps(before, indent=2) + "\n")
    with (logs / "build.log").open("w") as output:
        code = subprocess.call(command, stdout=output, stderr=subprocess.STDOUT)
    (logs / "exit.json").write_text(json.dumps(dict(returncode=code)) + "\n")
    assert all(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() == d for p, d in before.items()
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
