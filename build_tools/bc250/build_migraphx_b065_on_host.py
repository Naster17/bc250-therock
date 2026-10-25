#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build an isolated MIGraphX backend on durable SSHFS workspace storage."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import shutil
import time

from bc250_torch import runtime_lock
from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight

STORAGE = WS / "tmp/migraphx-b064/20261025-storage/mount/b065"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("abseil", "protobuf", "parser"), required=True
    )
    parser.add_argument("--attempt", default="r1")
    args = parser.parse_args()
    assert args.attempt in ("r1", "r2", "r3")
    preflight("migraphx-b065-" + args.phase)
    assert os.path.ismount(STORAGE.parent)
    fs = os.statvfs(STORAGE)
    available = fs.f_bavail * fs.f_frsize
    assert available >= 16 * 1024**3
    root_available = os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize
    assert root_available > 768 * 1024**2
    logs = STORAGE / ("logs-" + args.phase + "-" + args.attempt)
    logs.mkdir()
    shutil.copyfile(WS / "helpers/build_migraphx_b065.sh", logs / "build-script.sh")
    baseline = json.loads(
        (WS / "logs/migraphx-b064-qualification-r2/final.json").read_text()
    )
    assert baseline["status"] == "passed"
    command = [
        "podman",
        "run",
        "--name",
        "bc250-migraphx-b065-" + args.phase + "-" + args.attempt,
        "--network=none",
        "--read-only",
        "--cpus=4",
        "--memory=10g",
        "--memory-swap=12g",
    ]
    for src, dest, mode in (
        (WS / "source", "/workspace/source", "ro"),
        (WS / "build", "/workspace/build", "ro"),
        (WS / "helpers", "/workspace/helpers", "ro"),
        (WS / "migraphx-b062/rocm-cmake", "/workspace/deps", "ro"),
        (STORAGE, "/workspace/task", "rw"),
        (WS / "tmp/migraphx-b064/20261025-source-r4", "/workspace/task/source", "ro"),
        (WS / "tmp/migraphx-b065/abseil", "/workspace/abseil", "ro"),
        (logs, "/workspace/logs", "rw"),
    ):
        command += ["-v", f"{src}:{dest}:{mode}"]
    if args.phase != "abseil":
        command += ["-v", f"{WS / 'tmp/migraphx-b065/protobuf'}:/workspace/protobuf:ro"]
    command += [
        "fa3e8c254b6a",
        "bash",
        "/workspace/logs/build-script.sh",
        args.phase,
    ]
    report = dict(
        status="building",
        command=command,
        storage=str(STORAGE),
        available_bytes_before=available,
        root_available_bytes_before=root_available,
        root_floor_bytes=768 * 1024**2,
        build_and_cache_budget_bytes=14 * 1024**3,
        phase=args.phase,
        build_script_sha256=sha(logs / "build-script.sh"),
        runtime_changed=False,
        protected_sha256=baseline["protected_sha256"],
    )
    with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
        for path, digest in baseline["protected_sha256"].items():
            assert sha(Path(path)) == digest, path
        start = time.monotonic()
        try:
            with (logs / "controller.log").open("w") as stream:
                proc = subprocess.Popen(
                    command, stdout=stream, stderr=subprocess.STDOUT
                )
                report["controller_child_pid"] = proc.pid
                start = time.monotonic()
                atomic_json(logs / "progress.json", report)
                while proc.poll() is None:
                    time.sleep(10)
                    current = os.statvfs(STORAGE)
                    root_fs = os.statvfs(WS)
                    reason = None
                    if root_fs.f_bavail * root_fs.f_frsize < 768 * 1024**2:
                        reason = "Root rescue floor reached"
                    if current.f_bavail * current.f_frsize < available - 14 * 1024**3:
                        reason = "Durable build/cache budget reached"
                    if time.monotonic() - start > 7200:
                        reason = "Two-hour build bound reached"
                    if reason:
                        subprocess.run(
                            [
                                "podman",
                                "stop",
                                "--time",
                                "15",
                                "bc250-migraphx-b065-"
                                + args.phase
                                + "-"
                                + args.attempt,
                            ],
                            check=True,
                        )
                        proc.wait(timeout=30)
                        raise RuntimeError(reason)
                assert proc.returncode == 0, proc.returncode
            report["status"] = "built-isolated-not-qualified"
        except BaseException as error:
            report.update(status="failed", error=repr(error))
            raise
        finally:
            report["changed_hashes"] = [
                p for p, d in baseline["protected_sha256"].items() if sha(Path(p)) != d
            ]
            report["elapsed_seconds"] = time.monotonic() - start
            atomic_json(logs / "final.json", report)
    assert not report["changed_hashes"]


if __name__ == "__main__":
    main()
