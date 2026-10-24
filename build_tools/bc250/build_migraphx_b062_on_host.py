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
    parser.add_argument("--attempt", choices=("r1", "r2", "r3"), default="r2")
    args = parser.parse_args()
    preflight("migraphx-b062-build")
    assert os.geteuid() == 0
    root = WS / "migraphx-b062"
    source = root / "source"
    if not source.exists():
        source.mkdir()
        with tarfile.open(root / "source.tar.gz") as tar:
            tar.extractall(source, filter="data")
    logs = WS / ("logs/migraphx-b062-build-" + args.attempt)
    logs.mkdir()
    task = WS / ("tmp/migraphx-b062/20261024-" + args.attempt)
    task.mkdir(parents=True)
    (task / "source").symlink_to(source)
    baseline = json.loads((WS / "logs/onnx-b061-installed-r1/final.json").read_text())
    assert baseline["status"] == "passed"
    report = dict(
        status="building",
        source_archive_sha256=sha(root / "source.tar.gz"),
        free_bytes_before=shutil.disk_usage(WS).free,
        full_build_started=False,
        protected_sha256=baseline["protected_sha256"],
    )
    command = [
        "podman",
        "run",
        "--name",
        "bc250-migraphx-b062-build-" + args.attempt,
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
    command += ["fa3e8c254b6a", "bash", "/workspace/helpers/build_migraphx_b062.sh"]
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
            report.update(
                status="probes-built",
                configure_exit=int((logs / "configure-exit.txt").read_text()),
            )
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
