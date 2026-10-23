#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Preserve the missed-reduce-guard candidate and resume the exact source overlay."""

import json
import os
from pathlib import Path
import shutil
import subprocess

from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight

assert os.geteuid() == 0
preflight("rccl-b037-rebuild")
assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
baseline = json.loads((WS / "logs/rccl-b036-qualification-r1/final.json").read_text())
for name, expected in baseline["protected_sha256"].items():
    assert sha(Path(name)) == expected, name
assert shutil.disk_usage(WS).free > 7 * 1024**3
failed = json.loads((WS / "logs/rccl-b037-r1-guards/result.json").read_text())
assert failed["status"] == "failed" and len(failed["checks"]) == 33
old = json.loads((WS / "logs/rccl-b037-build/final.json").read_text())
assert old["status"] == "built-not-qualified"
candidate = Path(old["candidate"])
source = json.loads((WS / "helpers/b037-source-r2.json").read_text())
recovery = WS / "recovery/rccl-b037-reduce-before"
recovery.mkdir()
for name, expected in old["native_sha256"].items():
    p = candidate / "torch/lib" / name
    assert sha(p) == expected, name
    shutil.copyfile(p, recovery / name)
shutil.copyfile(candidate / "GIT_REVISION", recovery / "GIT_REVISION")
for name, expected in source["source_sha256"].items():
    target = recovery / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(candidate / name, target)
    overlay = WS / "helpers" / ("b037-" + Path(name).name)
    assert sha(overlay) == expected, name
    shutil.copyfile(overlay, candidate / name)
atomic_json(WS / "staging/rccl-b037/source-r2.json", source)
(candidate / "GIT_REVISION").write_text(source["torch_revision"] + "\n")
logs = WS / "logs/rccl-b037-build-r2"
logs.mkdir()
command = list(old["command"])
command[command.index(old["container"])] = "bc250-rccl-b037-build-r2"
command[command.index(str(WS / "logs/rccl-b037-build") + ":/workspace/logs:rw")] = (
    str(logs) + ":/workspace/logs:rw"
)
report = dict(old)
report.update(
    status="building",
    pid=os.getpid(),
    container="bc250-rccl-b037-build-r2",
    command=command,
    source=source,
)
atomic_json(logs / "progress.json", report)
try:
    with (logs / "build.log").open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    atomic_json(logs / "exit.json", dict(returncode=result.returncode))
    assert result.returncode == 0
    for name, expected in baseline["protected_sha256"].items():
        assert sha(Path(name)) == expected, name
    report.update(
        status="built-not-qualified",
        native_sha256={p.name: sha(p) for p in (candidate / "torch/lib").glob("*.so")},
        disk_available_bytes=shutil.disk_usage(WS).free,
    )
except BaseException as error:
    report.update(status="failed", error=repr(error))
    raise
finally:
    atomic_json(logs / "final.json", report)
