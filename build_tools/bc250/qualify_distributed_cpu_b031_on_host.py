#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the local Gloo gate in a bounded container without any GPU devices."""

import json
import os
from pathlib import Path
import shutil
import subprocess

from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS

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
assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
stage = json.loads((WS / "logs/distributed-b031-stage-r1/final.json").read_text())
source = Path(stage["candidate_torch"])
for name, expected in stage["native_sha256"].items():
    assert sha(source / "torch/lib" / name) == expected, name
baseline = json.loads((WS / "logs/sparse-b021-qualification-r5/final.json").read_text())
for path, expected in baseline["protected_sha256"].items():
    assert sha(Path(path)) == expected, path
active = Path("/opt/bc250-stack/current/runtime.json")
runtime = json.loads(active.read_text())
logs = WS / "logs/distributed-b031-cpu-r1"
logs.mkdir()
os.chown(logs, runtime["uid"], runtime["gid"])
command = [
    "podman",
    "run",
    "--rm",
    "--name",
    "bc250-distributed-b031-cpu-r1",
    "--network=host",
    "--cpus=2",
    "--memory=4g",
    "--memory-swap=5g",
    "--shm-size=1g",
    "--user",
    f"{runtime['uid']}:{runtime['gid']}",
]
for src, dst, mode in [
    (source, "/workspace/pytorch-work", "ro"),
    (WS / "helpers", "/workspace/helpers", "ro"),
    (WS / "build/dist/rocm", "/workspace/build/dist/rocm", "ro"),
    (logs, "/workspace/logs", "rw"),
]:
    command += ["-v", f"{src}:{dst}:{mode}"]
for item in (
    "PYTHONPATH=/workspace/pytorch-work",
    "PYTHONDONTWRITEBYTECODE=1",
    "PYTHONFAULTHANDLER=1",
    "OMP_NUM_THREADS=1",
    "OPENBLAS_NUM_THREADS=1",
    "LD_LIBRARY_PATH=/workspace/build/dist/rocm/lib:/workspace/build/dist/rocm/lib/llvm/lib",
    "PATH=/workspace/build/dist/rocm/bin:/workspace/build/dist/rocm/lib/llvm/bin:/opt/torch-venv/bin:/usr/bin:/bin",
):
    command += ["-e", item]
command += [
    runtime["image"],
    "python",
    "/workspace/helpers/validate_distributed_b031.py",
    "--output",
    "/workspace/logs/result.json",
]
assert "--device" not in command
before = subprocess.check_output(["dmesg"])
(logs / "kernel-before.log").write_bytes(before)
report = dict(
    status="running",
    command=command,
    gpu_devices_exposed=False,
    active_runtime_sha256=sha(active),
    source=stage["native_sha256"],
    boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    source_sha256=sha(WS / "helpers/validate_distributed_b031.py"),
)
atomic_json(logs / "before.json", report)
try:
    with (logs / "container.log").open("w") as stream:
        result = subprocess.run(
            command, stdout=stream, stderr=subprocess.STDOUT, timeout=240
        )
    report["returncode"] = result.returncode
    assert result.returncode == 0, result.returncode
    checks = json.loads((logs / "result.json").read_text())
    assert checks["status"] == "passed" and len(checks["checks"]) == 31
    assert sha(active) == report["active_runtime_sha256"]
    for path, expected in baseline["protected_sha256"].items():
        assert sha(Path(path)) == expected, path
    report.update(
        status="passed",
        checks=31,
        protected_paths=len(baseline["protected_sha256"]),
        disk_available_bytes=shutil.disk_usage(WS).free,
    )
except BaseException as error:
    report.update(status="failed", error=repr(error))
    raise
finally:
    after = subprocess.check_output(["dmesg"])
    (logs / "kernel-after.log").write_bytes(after)
    report["kernel_log_unchanged"] = before == after
    atomic_json(logs / "final.json", report)
