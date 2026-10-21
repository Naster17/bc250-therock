#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Activate qualified CPU audio codec image with exact runtime metadata rollback."""

import json
import os
from pathlib import Path
import subprocess
import sys

from deploy_torch_a04_on_host import atomic_json, sha
from qualify_video_b044_on_host import GATES, SOURCES
from qualify_training_n33_on_host import WS, main
from qualify_training_n39_on_host import preflight

assert os.geteuid() == 0
assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
active = Path("/opt/bc250-stack/current/runtime.json")
backup = WS / "recovery/video-b044-before/runtime.json"
assert backup.is_file()


def restore():
    temp = active.with_name("runtime.video-b044-restore.json")
    assert not temp.exists()
    temp.write_bytes(backup.read_bytes())
    with temp.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temp, active)
    assert sha(active) == sha(backup)


if sys.argv[1:] == ["--rollback"]:
    deployment = json.loads(
        (WS / "logs/video-b044-deployment-r1/final.json").read_text()
    )
    assert sha(active) == deployment["installed_runtime_sha256"]
    restore()
    print("CODEC_B043_ROLLBACK_PASS", sha(active))
    raise SystemExit(0)
assert not sys.argv[1:]
preflight("video-b044-install-r1")
qualified = json.loads((WS / "logs/video-b044-qualification-r2/final.json").read_text())
assert qualified["status"] == "passed"
assert [g["checks"] for g in qualified["gates"]] == [55, 46, 7]
for name, expected in qualified["protected_sha256"].items():
    assert sha(Path(name)) == expected, name
assert sha(active) == qualified["runtime_sha256"] == sha(backup)
stage = json.loads((WS / "logs/video-b044-stage-r1/final.json").read_text())
candidate = Path(stage["candidate_release"]) / "runtime.json"
assert sha(candidate) == stage["candidate_runtime_sha256"]
manifest = json.loads(candidate.read_text())
assert manifest["image"] == stage["image"]
manifest["torchcodec_b044"].update(
    scope="Bounded CPU lossless FFV1/H264-RGB video decode/seek/encode, exact VFR timestamp gap repair and HIP spatial forward/backward interoperability. CUDA/image paths off; approximate/VFR universality/performance unqualified.",
    candidate_evidence=str(WS / "logs/video-b044-qualification-r2/final.json"),
    previous_image=stage["previous_image"],
)
logs = WS / "logs/video-b044-deployment-r1"
logs.mkdir()
report = dict(
    status="installing",
    previous_runtime_sha256=sha(active),
    image=stage["image"],
    previous_image=stage["previous_image"],
    recovery=str(backup),
    prefix_changed=True,
    native_libraries_changed=False,
)
try:
    atomic_json(active, manifest)
    report["installed_runtime_sha256"] = sha(active)
    atomic_json(logs / "progress.json", report)
    baseline = WS / "logs/video-b044-installed-baseline-r1"
    baseline.mkdir()
    qualified["protected_sha256"][str(active)] = sha(active)
    qualified["runtime_sha256"] = sha(active)
    qualified["status"] = "installation-preservation-baseline"
    atomic_json(baseline / "final.json", qualified)
    sys.argv = [sys.argv[0], "--attempt", "r3"]
    main(
        batch_id="video-b044",
        include_training_app=False,
        baseline_label=baseline.name,
        default_phase="extra",
        extra_gates=GATES,
        additional_sources=SOURCES,
        minimum_disk_free_bytes=4 * 1024**3,
    )
    report.update(
        status="passed",
        installed_evidence=str(WS / "logs/video-b044-qualification-r3/final.json"),
        rollback_used=False,
    )
except BaseException as error:
    restore()
    report.update(
        status="failed",
        error=repr(error),
        rollback_used=True,
        restored_runtime_sha256=sha(active),
    )
    raise
finally:
    atomic_json(logs / "final.json", report)
