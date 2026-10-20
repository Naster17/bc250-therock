#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a new isolated image/wheel under the existing exclusive prefix lock."""

import json
import os
from pathlib import Path
import subprocess
import tarfile

from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS

if __name__ == "__main__":
    assert os.geteuid() == 0
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    prior = json.loads((WS / "logs/vision-b041-entry-r2/final.json").read_text())
    assert prior["status"] == "recorded-documentation-entry-drift"
    for name, expected in prior["protected_sha256"].items():
        assert sha(Path(name)) == expected, name
    root = WS / "vision-b041"
    assert os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize > 8 * 1024**3
    assert not (root / "source").exists()
    with tarfile.open(root / "source.tar") as t:
        t.extractall(root, filter="data")
    pillow = json.loads((root / "pillow-source.json").read_text())
    assert sha(root / pillow["filename"]) == pillow["sha256"]
    logs = WS / "logs/vision-b041-build-r2"
    logs.mkdir()
    build_image = "localhost/bc250-torchvision-build:b041-r2"
    assert subprocess.run(["podman", "image", "exists", build_image]).returncode != 0
    base = subprocess.check_output(
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
    assert base.startswith("28ad36e62398") or base.startswith(
        "sha256:28ad36e62398"
    ), base
    report = dict(
        status="building-image",
        base_image=base,
        source_revision="8fb87713a24951e639c494b0f2a8a81b5f8e33a6",
        source_archive_sha256=sha(root / "source.tar"),
        pillow=pillow,
        protected_sha256=prior["protected_sha256"],
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        build_resources=dict(
            jobs=4, memory_bytes=10 * 1024**3, ccache_max="20G", gpu_devices=False
        ),
        source_sha256={
            n: sha(WS / "helpers" / n)
            for n in (
                "build_vision_b041.sh",
                "build_vision_b041_on_host.py",
                "Containerfile.vision-b041",
            )
        },
    )
    atomic_json(logs / "progress.json", report)
    try:
        with (logs / "image.log").open("w") as log:
            subprocess.run(
                [
                    "podman",
                    "build",
                    "--pull=never",
                    "--network=host",
                    "--tag",
                    build_image,
                    "--file",
                    str(WS / "helpers/Containerfile.vision-b041"),
                    str(root / "image-context"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=900,
            )
        report["status"] = "building-wheel"
        atomic_json(logs / "progress.json", report)
        command = [
            "podman",
            "run",
            "--rm",
            "--name",
            "vision-b041-build-r2",
            "--network=none",
            "--memory=10g",
            "--memory-swap=12g",
            "--cpus=8",
            "--entrypoint",
            "bash",
        ]
        for source, target, mode in (
            (root, "/workspace/vision-b041", "rw"),
            (WS / "helpers", "/workspace/helpers", "ro"),
            (WS / "source", "/workspace/source", "ro"),
            (WS / "cache", "/workspace/cache", "rw"),
            (Path("/opt/bc250-stack/current/torch"), "/workspace/pytorch-work", "ro"),
            (Path("/opt/bc250-stack/current/rocm"), "/workspace/build/dist/rocm", "ro"),
        ):
            command += ["-v", f"{source}:{target}:{mode}"]
        command += [build_image, "/workspace/helpers/build_vision_b041.sh"]
        atomic_json(logs / "command.json", command)
        with (logs / "wheel.log").open("w") as log:
            subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=3600
            )
        wheels = list((root / "wheels").glob("torchvision-*.whl"))
        assert len(wheels) == 1, wheels
        report.update(
            status="passed",
            wheel=str(wheels[0]),
            wheel_sha256=sha(wheels[0]),
            build_image=subprocess.check_output(
                ["podman", "image", "inspect", build_image, "--format", "{{.Id}}"],
                text=True,
            ).strip(),
        )
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        report["changed_protected_hashes"] = [
            n for n, h in prior["protected_sha256"].items() if sha(Path(n)) != h
        ]
        assert not report["changed_protected_hashes"], report[
            "changed_protected_hashes"
        ]
        atomic_json(logs / "final.json", report)
