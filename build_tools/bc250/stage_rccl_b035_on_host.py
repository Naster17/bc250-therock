#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prepare an isolated RCCL Torch release and a stack-only preservation baseline."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", default="r1")
    parser.add_argument("--build-label", default="rccl-b035-build")
    args = parser.parse_args()
    assert re.fullmatch(r"r[1-9][0-9]*", args.attempt)
    assert re.fullmatch(r"rccl-b035-build(?:-r[1-9][0-9]*)?", args.build_label)
    assert os.geteuid() == 0
    preflight("rccl-b035-stage")
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    built = json.loads((WS / "logs" / args.build_label / "final.json").read_text())
    assert built["status"] == "built-not-qualified"
    baseline = json.loads(
        (WS / "logs/rccl-b034-qualification-r1/final.json").read_text()
    )
    assert baseline["status"] == "passed"
    protected = dict(baseline["protected_sha256"])
    for name, expected in protected.items():
        assert sha(Path(name)) == expected, name
    source = Path(built["candidate"])
    for name, expected in built["native_sha256"].items():
        path = source / "torch/lib" / name
        assert sha(path) == expected, path
        protected[str(path)] = expected
    for name in ("GIT_REVISION", "build/CMakeCache.txt"):
        protected[str(source / name)] = sha(source / name)
    active = Path("/opt/bc250-stack/current").resolve(strict=True)
    candidate = WS / "rccl-b035" / ("candidate-release-" + args.attempt)
    candidate.mkdir(parents=True)
    for entry in active.iterdir():
        if entry.name not in ("runtime.json", "torch"):
            (candidate / entry.name).symlink_to(entry)
    (candidate / "torch").symlink_to(source)
    manifest = json.loads((active / "runtime.json").read_text())
    manifest["torch_revision"] = built["source"]["torch_revision"]
    manifest["rccl_b035"] = dict(
        status="isolated candidate; qualification pending",
        source=built["source"],
        build_evidence=str(WS / "logs" / args.build_label / "final.json"),
        rccl=str(WS / "rccl-b034/install-r1"),
        scope="one physical GPU/rank1; not installed; no multi-GPU/transport/scaling proof",
    )
    atomic_json(candidate / "runtime.json", manifest)
    protected[str(candidate / "runtime.json")] = sha(candidate / "runtime.json")
    baseline["protected_sha256"] = protected
    logs = WS / "logs" / ("rccl-b035-baseline-" + args.attempt)
    logs.mkdir()
    atomic_json(logs / "final.json", baseline)
    recovery = WS / "recovery/rccl-b035-before"
    recovery.mkdir(exist_ok=True)
    if args.attempt == "r1":
        shared = WS / "helpers/qualify_training_n33_on_host.py"
        shutil.copyfile(shared, recovery / shared.name)
        (recovery / "runtime.json").write_bytes((active / "runtime.json").read_bytes())
        (recovery / "torch-link.txt").write_text(os.readlink(active / "torch") + "\n")
        os.replace(WS / "helpers/qualify_training_n33.rccl-b035-next.py", shared)
    stage = WS / "logs" / ("rccl-b035-stage-" + args.attempt)
    stage.mkdir()
    atomic_json(
        stage / "final.json",
        dict(
            status="passed",
            candidate_release=str(candidate),
            candidate_runtime_sha256=sha(candidate / "runtime.json"),
            active_runtime_sha256=sha(active / "runtime.json"),
            protected_paths=len(protected),
        ),
    )


if __name__ == "__main__":
    main()
