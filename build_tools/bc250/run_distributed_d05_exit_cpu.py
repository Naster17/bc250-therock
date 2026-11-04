#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run CPU lifecycle checks in the pinned daily image with bounded resources."""
import json
import os
from pathlib import Path
import pwd
import sys
import tempfile

sys.path.insert(0, "/opt/bc250-stack/tools/distributed-d01")
from bc250_torch import LaunchOptions, build_command, runtime_lock, run_process

assert os.geteuid() == 0 and os.uname().nodename == "bc250"
release = Path("/opt/bc250-stack/current").resolve()
manifest = json.loads((release / "runtime.json").read_text())
source = Path("/home/nik/bc250work/tools/distributed-d05/exit-source")
base = Path("/home/nik/bc250work/evidence/distributed-d05-exit-cpu-r2")
base.mkdir()
os.chown(base, 1000, 1000)
output = base / "tests"
options = LaunchOptions(
    arguments=[
        str(source / "distributed_d05_exit_store.py"),
        "--candidate",
        str(source / "api.py"),
        "--unit-module",
        str(source / "api_test.py"),
        "--output",
        str(output),
    ],
    environment={"OMP_NUM_THREADS": "1"},
)
account = pwd.getpwuid(manifest["uid"])
groups = os.getgrouplist(account.pw_name, manifest["gid"]) + [
    Path(x).stat().st_gid for x in ["/dev/kfd", "/dev/dri/renderD128"]
]
with runtime_lock(release / ".runtime.lock"):
    with tempfile.TemporaryDirectory() as folder:
        envfile = Path(folder) / "env"
        envfile.write_text("OMP_NUM_THREADS=1\n")
        command = build_command(
            release,
            manifest,
            Path(account.pw_dir),
            Path(account.pw_dir),
            options,
            groups,
            envfile,
        )
        command[2:2] = [
            "--cpus=4",
            "--cpuset-cpus=0-3",
            "--memory=4g",
            "--memory-swap=6g",
        ]
        raise SystemExit(run_process(command))
