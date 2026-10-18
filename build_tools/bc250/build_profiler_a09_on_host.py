#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build an isolated Kineto candidate without changing the active runtime."""
import json
import os
from pathlib import Path
import subprocess

WS = Path("/home/nik/bc250work/rocm-10.2-gfx1013")
CANDIDATE = WS / "pytorch-profiler-a09"


def main() -> None:
    assert os.geteuid() == 0
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    source = (Path("/opt/bc250-stack/current") / "torch").resolve()
    assert source == WS / "pytorch-sparse-followup"
    assert (
        source / "GIT_REVISION"
    ).read_text().strip() == "71a0426566bc9b952d9aa9b41f4a9d79ce46e0e8"
    assert not CANDIDATE.exists()
    CANDIDATE.mkdir()
    # Tar retains internal hardlinks while omitting immutable shared dependencies.
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
    logs = WS / "logs/profiler-a09-build"
    logs.mkdir()
    command = [
        "podman",
        "run",
        "--name",
        "bc250-profiler-a09-build",
        "--network",
        "none",
    ]
    for src, dst, mode in [
        (WS / "source", "/workspace/source", "ro"),
        (WS / "helpers", "/workspace/helpers", "ro"),
        (CANDIDATE, "/workspace/pytorch-work", "rw"),
        (source / "third_party", "/workspace/pytorch-work/third_party", "ro"),
        (WS / "build", "/workspace/build", "ro"),
        (WS / "cache", "/workspace/cache", "rw"),
        (logs, "/workspace/logs", "rw"),
    ]:
        command += ["-v", f"{src}:{dst}:{mode}"]
    command += [
        "localhost/bc250-pytorch-build:2.13-gfx1013",
        "bash",
        "/workspace/helpers/build_torch_a09.sh",
    ]
    (logs / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (logs / "build.log").open("w") as output:
        code = subprocess.call(command, stdout=output, stderr=subprocess.STDOUT)
    (logs / "exit.json").write_text(json.dumps({"returncode": code}) + "\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
