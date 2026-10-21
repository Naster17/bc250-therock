#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Activate qualified audio/HIP fix; roll back image, DSO and source together."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from deploy_torch_a04_on_host import atomic_json, check_gate, sha
from qualify_training_n33_on_host import WS, main
from qualify_training_n39_on_host import preflight

assert os.geteuid() == 0
assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
active = Path("/opt/bc250-stack/current/runtime.json")
recovery = WS / "recovery/audio-b042-before"
backup = recovery / "runtime.json"
library = Path("/opt/bc250-stack/current/rocm/lib/libamdhip64.so").resolve(strict=True)
source = WS / "source/rocm-systems/projects/clr/rocclr/device/rocm/rocvirtual.cpp"
assert backup.is_file()


def atomic_copy(src, dest):
    temporary = dest.with_name(dest.name + ".audio-b042-tmp")
    assert not temporary.exists()
    attributes = dest.stat()
    shutil.copy2(src, temporary)
    os.chown(temporary, attributes.st_uid, attributes.st_gid)
    os.chmod(temporary, attributes.st_mode & 0o777)
    os.replace(temporary, dest)
    assert sha(src) == sha(dest)


def restore():
    for saved, target in (
        (recovery / "libamdhip64.original.so", library),
        (recovery / "rocvirtual.original.cpp", source),
        (backup, active),
    ):
        atomic_copy(saved, target)


if sys.argv[1:] == ["--rollback"]:
    installed = json.loads(
        (WS / "logs/audio-b042-deployment-r1/final.json").read_text()
    )
    assert sha(active) == installed["installed_runtime_sha256"]
    assert sha(library) == installed["installed_library_sha256"]
    assert sha(source) == installed["installed_source_sha256"]
    restore()
    print("AUDIO_B042_ROLLBACK_PASS", sha(active), sha(library), sha(source))
    raise SystemExit(0)

assert not sys.argv[1:]
preflight("audio-b042-install-r1")
qualified = json.loads((WS / "logs/audio-b042-qualification-r4/final.json").read_text())
assert qualified["status"] == "passed"
assert [g["checks"] for g in qualified["gates"]] == [34, 40, 51, 9, 7]
for name, expected in qualified["protected_sha256"].items():
    assert sha(Path(name)) == expected, name
assert sha(active) == qualified["runtime_sha256"] == sha(backup)
assert sha(source) == "c87d34b1cbba30226bdace35653a12e8c0d6e29afc606db3fc1c2d7fc917685c"
check_gate("audio-b042-hostcall-candidate-r1")
probe = json.loads(
    (WS / "logs/audio-b042-hostcall-candidate-r1/result.json").read_text()
)
assert probe["status"] == "passed" and len(probe["checks"]) == 1
candidate = WS / "audio-b042/runtime-candidate-r1/libamdhip64.so.7.17.0-0000000"
assert sha(candidate) == qualified["hip_candidate_sha256"]
build = json.loads((WS / "audio-b042/runtime-candidate-r1/final.json").read_text())
assert sha(candidate) == build["candidate_sha256"]
assert sha(library) == build["original_sha256"]
assert sha(WS / "audio-b042/rocvirtual.cpp") == build["source_sha256"]
for saved, target in (
    (recovery / "libamdhip64.original.so", library),
    (recovery / "rocvirtual.original.cpp", source),
):
    assert not saved.exists()
    shutil.copy2(target, saved)
    assert sha(saved) == sha(target)
stage = json.loads((WS / "logs/audio-b042-stage-r2/final.json").read_text())
candidate_manifest = Path(stage["candidate_release"]) / "runtime.json"
assert sha(candidate_manifest) == stage["candidate_runtime_sha256"]
manifest = json.loads(candidate_manifest.read_text())
manifest["rocm_systems_revision"] = "1f9609fd69c4923291f58934495520fe62e66ef8"
manifest["torchaudio_b042"].update(
    scope="Bounded native HIP IIR FP32/FP64 output/all-gradient, RNNT FP16/FP32 loss/gradient, forced alignment and audio transforms. File load/save requires separately uninstalled TorchCodec. No universal correctness/performance claim.",
    candidate_evidence=str(WS / "logs/audio-b042-qualification-r4/final.json"),
    previous_image=stage["previous_image"],
)
manifest["clr_hostcall_b042"] = dict(
    source_revision=manifest["rocm_systems_revision"],
    library_sha256=sha(candidate),
    source_sha256=build["source_sha256"],
    scope="Protect/reacquire the internal transfer HW queue during cold hostcall occupied-bitfield initialization.",
    candidate_probe=str(WS / "logs/audio-b042-hostcall-candidate-r1"),
    kernel_packs_changed=False,
    recovery=str(recovery),
)
logs = WS / "logs/audio-b042-deployment-r1"
logs.mkdir()
report = dict(
    status="installing",
    previous_runtime_sha256=sha(active),
    previous_library_sha256=sha(library),
    previous_source_sha256=sha(source),
    image=stage["image"],
    previous_image=stage["previous_image"],
    recovery=str(recovery),
    prefix_changed=True,
)
try:
    atomic_copy(candidate, library)
    atomic_copy(WS / "audio-b042/rocvirtual.cpp", source)
    atomic_json(active, manifest)
    report.update(
        installed_runtime_sha256=sha(active),
        installed_library_sha256=sha(library),
        installed_source_sha256=sha(source),
    )
    atomic_json(logs / "progress.json", report)
    for name in qualified["protected_sha256"]:
        if Path(name).resolve() == library or name == str(active):
            qualified["protected_sha256"][name] = sha(Path(name))
    qualified["protected_sha256"][str(source)] = sha(source)
    qualified["runtime_sha256"] = sha(active)
    qualified["status"] = "installation-preservation-baseline"
    baseline = WS / "logs/audio-b042-installed-baseline-r1"
    baseline.mkdir()
    atomic_json(baseline / "final.json", qualified)
    sys.argv = [sys.argv[0], "--attempt", "r5"]
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
            ("hostcall", "validate_hostcall_queue_b042.py", [], 1),
        ),
        additional_sources=(
            "validate_audio_b042.py",
            "validate_vision_b041.py",
            "validate_cleanup_runtime.py",
            "validate_hostcall_queue_b042.py",
        ),
        minimum_disk_free_bytes=4 * 1024**3,
    )
    report.update(
        status="passed",
        installed_evidence=str(WS / "logs/audio-b042-qualification-r5/final.json"),
        rollback_used=False,
    )
except BaseException as error:
    restore()
    report.update(
        status="failed",
        error=repr(error),
        rollback_used=True,
        restored_runtime_sha256=sha(active),
        restored_library_sha256=sha(library),
        restored_source_sha256=sha(source),
    )
    raise
finally:
    atomic_json(logs / "final.json", report)
