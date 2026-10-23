#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Preserve the failed BF16 candidate, apply the exact source fix and resume pip."""

import json
import os
from pathlib import Path
import shutil
import subprocess

from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight


def main():
    assert os.geteuid() == 0
    preflight("rccl-b035-rebuild")
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    baseline = json.loads(
        (WS / "logs/rccl-b034-qualification-r1/final.json").read_text()
    )
    for name, expected in baseline["protected_sha256"].items():
        assert sha(Path(name)) == expected, name
    assert shutil.disk_usage(WS).free > 7 * 1024**3
    failed = json.loads((WS / "logs/rccl-b035-r1-torch-rccl/result.json").read_text())
    assert failed["status"] == "failed" and len(failed["checks"]) == 31
    old = json.loads((WS / "logs/rccl-b035-build/final.json").read_text())
    assert old["status"] == "built-not-qualified"
    candidate = Path(old["candidate"])
    staged = WS / "staging/rccl-b035"
    for name in ("bf16-fix.json", "ProcessGroupNCCL.cpp"):
        shutil.copyfile(WS / "helpers" / ("b035-" + name), staged / name)
    overlay = json.loads((staged / "bf16-fix.json").read_text())
    path = candidate / "torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp"
    recovery = WS / "recovery/rccl-b035-bf16-before"
    recovery.mkdir()
    shutil.copyfile(path, recovery / path.name)
    (recovery / "GIT_REVISION").write_bytes((candidate / "GIT_REVISION").read_bytes())
    for name, expected in old["native_sha256"].items():
        p = candidate / "torch/lib" / name
        assert sha(p) == expected, name
        shutil.copyfile(p, recovery / name)
    atomic_json(recovery / "native-sha256.json", old["native_sha256"])
    assert sha(staged / path.name) == overlay["after_sha256"]
    shutil.copyfile(staged / path.name, path)
    (candidate / "GIT_REVISION").write_text(overlay["torch_revision"] + "\n")
    logs = WS / "logs/rccl-b035-build-r2"
    logs.mkdir()
    command = list(old["command"])
    command[command.index(old["container"])] = "bc250-rccl-b035-build-r2"
    command[command.index(str(WS / "logs/rccl-b035-build") + ":/workspace/logs:rw")] = (
        str(logs) + ":/workspace/logs:rw"
    )
    report = dict(old)
    report.update(
        status="building",
        pid=os.getpid(),
        container="bc250-rccl-b035-build-r2",
        command=command,
        source={
            **old["source"],
            "torch_revision": overlay["torch_revision"],
            "bf16_fix": overlay,
        },
    )
    atomic_json(logs / "progress.json", report)
    try:
        with (logs / "build.log").open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        atomic_json(logs / "exit.json", dict(returncode=result.returncode))
        assert result.returncode == 0, result.returncode
        for name, expected in baseline["protected_sha256"].items():
            assert sha(Path(name)) == expected, name
        report.update(
            status="built-not-qualified",
            native_sha256={
                p.name: sha(p) for p in (candidate / "torch/lib").glob("*.so")
            },
            disk_available_bytes=shutil.disk_usage(WS).free,
        )
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        atomic_json(logs / "final.json", report)


if __name__ == "__main__":
    main()
