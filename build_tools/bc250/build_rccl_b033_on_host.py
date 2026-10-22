#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build an independent filtered native RCCL candidate, preserving the active stack."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", default="r1")
    args = parser.parse_args()
    assert re.fullmatch(r"r[1-9][0-9]*", args.attempt)
    assert os.geteuid() == 0
    preflight("rccl-b033-build")
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    assert shutil.disk_usage(WS).free > 14 * 1024**3
    baseline = json.loads(
        (WS / "logs/distributed-b032-qualification-r1/final.json").read_text()
    )
    for path, expected in baseline["protected_sha256"].items():
        assert sha(Path(path)) == expected, path
    staged = WS / "staging/rccl-b033"
    source = json.loads((staged / "source.json").read_text())
    assert sha(staged / "rccl-source.tar") == source["source_archive_sha256"]
    candidate = WS / "rccl-b033"
    if args.attempt == "r1":
        candidate.mkdir()
        with tarfile.open(staged / "rccl-source.tar") as archive:
            archive.extractall(candidate / "source", filter="data")
    else:
        assert (candidate / "build-r1/build.ninja").is_file()
        overlay = json.loads((staged / "source-overlay.json").read_text())
        backup = WS / "recovery/rccl-b033-source-before"
        for name, metadata in overlay["files"].items():
            target = candidate / "source" / name
            payload = staged / metadata["staged"]
            assert sha(payload) == metadata["after_sha256"]
            if sha(target) == metadata["after_sha256"]:
                continue
            assert sha(target) == metadata["before_sha256"], name
            original = backup / name
            original.parent.mkdir(parents=True, exist_ok=True)
            assert not original.exists()
            shutil.copy2(target, original)
            target.write_bytes(payload.read_bytes())
        source["overlay"] = overlay
    logs = WS / ("logs/rccl-b033-build-" + args.attempt)
    logs.mkdir()
    image = subprocess.check_output(
        [
            "podman",
            "image",
            "inspect",
            "localhost/bc250-pytorch-build:2.13-gfx1013",
            "--format",
            "{{.Id}}",
        ],
        text=True,
    ).strip()
    container = "bc250-rccl-b033-build-" + args.attempt
    command = [
        "podman",
        "run",
        "--name",
        container,
        "--network=none",
        "--cpus=4",
        "--memory=10g",
        "--memory-swap=12g",
    ]
    for src, dst, mode in [
        (WS / "source", "/workspace/source", "ro"),
        (WS / "build", "/workspace/build", "ro"),
        (WS / "helpers", "/workspace/helpers", "ro"),
        (WS / "cache", "/workspace/cache", "rw"),
        (candidate, "/workspace/rccl", "rw"),
    ]:
        command += ["-v", f"{src}:{dst}:{mode}"]
    if args.attempt != "r1":
        command += ["-e", "RCCL_B033_RESUME=1"]
    command += [image, "bash", "/workspace/helpers/build_rccl_b033.sh"]
    report = dict(
        status="building",
        pid=os.getpid(),
        container=container,
        source=source,
        image=image,
        command=command,
        protected_sha256=baseline["protected_sha256"],
        active_prefix_changed=False,
    )
    atomic_json(logs / "progress.json", report)
    try:
        with (logs / "build.log").open("w") as stream:
            result = subprocess.run(
                command, stdout=stream, stderr=subprocess.STDOUT, timeout=2400
            )
        report["returncode"] = result.returncode
        assert result.returncode == 0, result.returncode
        report["status"] = "built-not-qualified"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        subprocess.run(
            ["podman", "stop", "--time", "10", container],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        raise
    finally:
        report["changed_hashes"] = [
            path
            for path, expected in baseline["protected_sha256"].items()
            if sha(Path(path)) != expected
        ]
        report["disk_available_bytes"] = shutil.disk_usage(WS).free
        atomic_json(logs / "final.json", report)
    assert not report["changed_hashes"]


if __name__ == "__main__":
    main()
