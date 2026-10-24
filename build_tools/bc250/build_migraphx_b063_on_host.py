#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Preserve active prefixes while auditing isolated MIGraphX build admission."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tarfile
import shutil

from bc250_torch import runtime_lock
from deploy_torch_a04_on_host import sha, atomic_json
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", choices=("r1", "r2", "r3", "r4"), default="r4")
    args = parser.parse_args()
    preflight("migraphx-b063-build")
    assert os.geteuid() == 0
    fs = os.statvfs(WS)
    budget = dict(
        ordinary_available_bytes=fs.f_bavail * fs.f_frsize,
        free_including_reserved_bytes=fs.f_bfree * fs.f_frsize,
        planned_build_limit_bytes=12 * 1024**3,
        planned_cache_growth_bytes=2 * 1024**3,
        ordinary_rescue_floor_bytes=2 * 1024**3,
        reserve_policy_changed=False,
        measured_full_build_footprint=False,
    )
    budget["full_build_admitted"] = budget["ordinary_available_bytes"] >= 16 * 1024**3
    assert (
        budget["ordinary_available_bytes"] >= 768 * 1024**2
    ), "Insufficient headroom even for bounded probes"
    root = WS / "migraphx-b063"
    source = root / "source"
    if not source.exists():
        source.mkdir()
        with tarfile.open(root / "source.tar.gz") as tar:
            tar.extractall(source, filter="data")
    logs = WS / ("logs/migraphx-b063-build-" + args.attempt)
    logs.mkdir()
    task = WS / ("tmp/migraphx-b063/20261024-" + args.attempt)
    task.mkdir(parents=True)
    (task / "source").symlink_to(source)
    baseline = json.loads((WS / "logs/onnx-b061-installed-r1/final.json").read_text())
    assert baseline["status"] == "passed"
    report = dict(
        status="building",
        disk_budget=budget,
        source_archive_sha256=sha(root / "source.tar.gz"),
        free_bytes_before=shutil.disk_usage(WS).free,
        full_build_started=False,
        protected_sha256=baseline["protected_sha256"],
    )
    command = [
        "podman",
        "run",
        "--name",
        "bc250-migraphx-b063-build-" + args.attempt,
        "--network=none",
        "--cpus=4",
        "--memory=10g",
        "--memory-swap=12g",
    ]
    for src, dst, mode in (
        (WS / "source", "/workspace/source", "ro"),
        (WS / "build", "/workspace/build", "ro"),
        (WS / "helpers", "/workspace/helpers", "ro"),
        (WS / "cache", "/workspace/cache", "rw"),
        (source, "/workspace/task/source", "ro"),
        (task, "/workspace/task", "rw"),
        (logs, "/workspace/logs", "rw"),
    ):
        command += ["-v", f"{src}:{dst}:{mode}"]
    command += ["fa3e8c254b6a", "bash", "/workspace/helpers/build_migraphx_b063.sh"]
    report["command"] = command
    atomic_json(logs / "progress.json", report)
    with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
        for name, digest in baseline["protected_sha256"].items():
            assert sha(Path(name)) == digest, name
        try:
            with (logs / "build.log").open("w") as stream:
                subprocess.run(
                    command,
                    check=True,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=240,
                )
            report.update(status="probes-built")
        except BaseException as error:
            report.update(status="failed", error=repr(error))
            raise
        finally:
            report["changed_hashes"] = [
                n for n, d in baseline["protected_sha256"].items() if sha(Path(n)) != d
            ]
            report["free_bytes_after"] = shutil.disk_usage(WS).free
            atomic_json(logs / "final.json", report)
    assert not report["changed_hashes"]


if __name__ == "__main__":
    main()
