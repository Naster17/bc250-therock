#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Install a qualified independent Gloo Torch build with exact link/metadata recovery."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from deploy_torch_a04_on_host import atomic_json, sha
from qualify_distributed_b031_on_host import GATES, SOURCES
from qualify_training_n33_on_host import WS, main
from qualify_training_n39_on_host import preflight

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--rollback", action="store_true")
parser.add_argument("--code-revision")
args = parser.parse_args()
assert os.geteuid() == 0
assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
active = Path("/opt/bc250-stack/current").resolve(strict=True)
recovery = WS / "recovery/distributed-b031-before"
old_runtime = recovery / "runtime.json"
old_target = (recovery / "torch-link.txt").read_text().strip()


def restore():
    temp = active / "torch.distributed-b031-restore"
    assert not temp.exists()
    temp.symlink_to(old_target)
    os.replace(temp, active / "torch")
    temporary = active / "runtime.distributed-b031-restore.json"
    assert not temporary.exists()
    temporary.write_bytes(old_runtime.read_bytes())
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, active / "runtime.json")
    assert sha(active / "runtime.json") == sha(old_runtime)
    assert os.readlink(active / "torch") == old_target


if args.rollback:
    assert args.code_revision is None
    deployed = json.loads(
        (WS / "logs/distributed-b031-deployment-r1/final.json").read_text()
    )
    assert sha(active / "runtime.json") == deployed["installed_runtime_sha256"]
    assert os.readlink(active / "torch") == deployed["torch_target"]
    restore()
    print("DISTRIBUTED_B031_ROLLBACK_PASS", sha(active / "runtime.json"))
    raise SystemExit(0)
assert args.code_revision is not None and re.fullmatch(
    "[0-9a-f]{40}", args.code_revision
)
preflight("distributed-b031-install")
qualified = json.loads(
    (WS / "logs/distributed-b031-qualification-r1/final.json").read_text()
)
assert qualified["status"] == "passed"
assert [g["checks"] for g in qualified["gates"]] == [g[3] for g in GATES]
for path, expected in qualified["protected_sha256"].items():
    assert sha(Path(path)) == expected, path
assert sha(active / "runtime.json") == sha(old_runtime) == qualified["runtime_sha256"]
assert os.readlink(active / "torch") == old_target
stage = json.loads((WS / "logs/distributed-b031-stage-r1/final.json").read_text())
candidate = Path(stage["candidate_release"])
assert sha(candidate / "runtime.json") == stage["candidate_runtime_sha256"]
source = Path(stage["candidate_torch"])
for name, expected in stage["native_sha256"].items():
    assert sha(source / "torch/lib" / name) == expected, name
manifest = json.loads((candidate / "runtime.json").read_text())
manifest["effective_therock_port_revision"] = args.code_revision
manifest[
    "source_state"
] += "; B03.1 local CPU Gloo distributed build, RCCL/MPI off; GPU distributed unqualified"
manifest["distributed_b031"].update(
    status="candidate-qualified; installed outcome in deployment evidence",
    installed_evidence=str(WS / "logs/distributed-b031-qualification-r2/final.json"),
    deployment_evidence=str(WS / "logs/distributed-b031-deployment-r1/final.json"),
    integration_revision=args.code_revision,
    candidate_evidence=str(WS / "logs/distributed-b031-qualification-r1/final.json"),
    previous_torch_target=old_target,
)
logs = WS / "logs/distributed-b031-deployment-r1"
logs.mkdir()
report = dict(
    status="installing",
    previous_runtime_sha256=sha(old_runtime),
    previous_torch_target=old_target,
    torch_target=str(source),
    image=manifest["image"],
    image_changed=False,
    rocm_libraries_changed=False,
    torch_libraries_changed=True,
    recovery=str(recovery),
    integration_revision=args.code_revision,
)
try:
    temp = active / "torch.distributed-b031-next"
    assert not temp.exists()
    temp.symlink_to(source)
    os.replace(temp, active / "torch")
    atomic_json(active / "runtime.json", manifest)
    report["installed_runtime_sha256"] = sha(active / "runtime.json")
    atomic_json(logs / "progress.json", report)
    baseline = WS / "logs/distributed-b031-installed-baseline-r1"
    baseline.mkdir()
    qualified["protected_sha256"][str(old_runtime)] = sha(old_runtime)
    qualified["protected_sha256"]["/opt/bc250-stack/current/runtime.json"] = sha(
        active / "runtime.json"
    )
    for p in [
        *(source / "torch/lib").glob("*.so"),
        *(source / "torch").glob("_C*.so"),
        source / "GIT_REVISION",
    ]:
        qualified["protected_sha256"][str(p)] = sha(p)
    qualified["runtime_sha256"] = sha(active / "runtime.json")
    qualified["torch_target"] = str(source)
    qualified["links"]["/opt/bc250-stack/current/torch"] = str(source)
    qualified["status"] = "installation-preservation-baseline"
    atomic_json(baseline / "final.json", qualified)
    sys.argv = [sys.argv[0], "--attempt", "r2"]
    main(
        batch_id="distributed-b031",
        baseline_label=baseline.name,
        include_training_app=False,
        default_phase="extra",
        extra_gates=GATES,
        additional_sources=SOURCES,
        minimum_disk_free_bytes=4 * 1024**3,
    )
    report.update(
        status="passed",
        installed_evidence=str(
            WS / "logs/distributed-b031-qualification-r2/final.json"
        ),
        rollback_used=False,
    )
except BaseException as error:
    restore()
    report.update(
        status="failed",
        error=repr(error),
        rollback_used=True,
        restored_runtime_sha256=sha(active / "runtime.json"),
    )
    raise
finally:
    atomic_json(logs / "final.json", report)
