#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Activate only the qualified audio image, with exact runtime-JSON rollback."""

import json
import os
from pathlib import Path
import subprocess
import sys

from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS, main
from qualify_training_n39_on_host import preflight

assert os.geteuid() == 0
assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
active = Path("/opt/bc250-stack/current/runtime.json")
backup = WS / "recovery/audio-b042-before/runtime.json"
assert backup.is_file()


def restore():
    temporary = active.with_name("runtime.audio-b042-restore.json")
    assert not temporary.exists()
    temporary.write_bytes(backup.read_bytes())
    os.replace(temporary, active)
    assert sha(active) == sha(backup)


if sys.argv[1:] == ["--rollback"]:
    deployment = json.loads(
        (WS / "logs/audio-b042-deployment-r1/final.json").read_text()
    )
    assert sha(active) == deployment["installed_runtime_sha256"]
    restore()
    print("AUDIO_B042_ROLLBACK_PASS", sha(active))
    raise SystemExit(0)

assert not sys.argv[1:]
preflight("audio-b042-install-r1")
qualified = json.loads((WS / "logs/audio-b042-qualification-r2/final.json").read_text())
assert qualified["status"] == "passed"
assert [(g["checks"]) for g in qualified["gates"]] == [34, 40, 51, 9, 7]
for name, expected in qualified["protected_sha256"].items():
    assert sha(Path(name)) == expected, name
assert sha(active) == qualified["runtime_sha256"] == sha(backup)
stage = json.loads((WS / "logs/audio-b042-stage-r2/final.json").read_text())
candidate = Path(stage["candidate_release"]) / "runtime.json"
assert sha(candidate) == stage["candidate_runtime_sha256"]
manifest = json.loads(candidate.read_text())
assert manifest["image"] == stage["image"]
manifest["torchaudio_b042"].update(
    scope="Bounded native HIP IIR FP32/FP64 output/all-gradient, RNNT FP16/FP32 loss/gradient, forced alignment and audio transforms. File load/save requires separately uninstalled TorchCodec. No universal correctness/performance claim.",
    candidate_evidence=str(WS / "logs/audio-b042-qualification-r2/final.json"),
    raw_wheel_sha256=json.loads(
        (WS / "logs/audio-b042-build-r4/final.json").read_text()
    )["wheel_sha256"],
    previous_image=stage["previous_image"],
)
logs = WS / "logs/audio-b042-deployment-r1"
logs.mkdir()
report = dict(
    status="installing",
    previous_runtime_sha256=sha(active),
    image=stage["image"],
    previous_image=stage["previous_image"],
    recovery=str(backup),
    prefix_changed=True,
)
try:
    atomic_json(active, manifest)
    report["installed_runtime_sha256"] = sha(active)
    atomic_json(logs / "progress.json", report)
    baseline = WS / "logs/audio-b042-installed-baseline-r1"
    baseline.mkdir()
    qualified["protected_sha256"][str(active)] = sha(active)
    qualified["runtime_sha256"] = sha(active)
    qualified["status"] = "installation-preservation-baseline"
    atomic_json(baseline / "final.json", qualified)
    sys.argv = [sys.argv[0], "--attempt", "r3"]
    main(
        batch_id="audio-b042",
        baseline_label=baseline.name,
        default_phase="extra",
        extra_gates=(
            ("dsp", "validate_audio_b042.py", ["--group", "dsp"], 34),
            ("native", "validate_audio_b042.py", ["--group", "native"], 40),
            ("vision-ops", "validate_vision_b041.py", ["--group", "ops"], 51),
            (
                "vision-integration",
                "validate_vision_b041.py",
                ["--group", "integration"],
                9,
            ),
            ("core", "validate_cleanup_runtime.py", [], 7),
        ),
        additional_sources=(
            "validate_audio_b042.py",
            "validate_vision_b041.py",
            "validate_cleanup_runtime.py",
        ),
        minimum_disk_free_bytes=4 * 1024**3,
    )
    report.update(
        status="passed",
        installed_evidence=str(WS / "logs/audio-b042-qualification-r3/final.json"),
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
