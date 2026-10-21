#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build dependencies, CPU codec wheel and candidate image under prefix lock."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("phase", choices=("deps", "build", "stage"))
parser.add_argument(
    "--attempt", default="r1", choices=("r1", "r2", "r3", "r4", "r5", "r6")
)
args = parser.parse_args()
assert os.geteuid() == 0
assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
preflight("codec-b043-" + args.phase + "-" + args.attempt)
root = WS / "codec-b043"
root.mkdir(exist_ok=True)
logs = WS / ("logs/codec-b043-" + args.phase + "-" + args.attempt)
logs.mkdir()
baseline = json.loads((WS / "logs/codec-b043-entry-r4/final.json").read_text())
protected = baseline["protected_sha256"]
for name, expected in protected.items():
    assert sha(Path(name)) == expected, name
assert os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize > 4 * 1024**3
report = dict(
    status="running",
    protected_sha256=protected,
    boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
)
(root / "tmp").mkdir(exist_ok=True)
os.environ["TMPDIR"] = str(root / "tmp")


def run(command, filename, timeout=1800):
    atomic_json(logs / (filename + ".command.json"), command)
    with (logs / (filename + ".log")).open("w") as stream:
        subprocess.run(
            command,
            check=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )


def image(tag):
    return subprocess.check_output(
        ["podman", "image", "inspect", tag, "--format", "{{.Id}}"], text=True
    ).strip()


try:
    if args.phase == "deps":
        context = root / "deps-context"
        context.mkdir()
        run(
            [
                "podman",
                "build",
                "--pull=never",
                "--memory=8g",
                "--memory-swap=10g",
                "--tag",
                "localhost/bc250-codec-build:b043-r1",
                "--file",
                str(WS / "helpers/Containerfile.codec-build-b043"),
                str(context),
            ],
            "image",
        )
        report["image"] = image("localhost/bc250-codec-build:b043-r1")
    elif args.phase == "build":
        if not (root / "source").exists():
            (root / "source").mkdir()
            with tarfile.open(root / "source.tar") as archive:
                archive.extractall(root / "source", filter="data")
        deps = json.loads((WS / "logs/codec-b043-deps-r2/final.json").read_text())
        assert deps["status"] == "passed"
        command = [
            "podman",
            "run",
            "--rm",
            "--name",
            "codec-b043-build-" + args.attempt,
            "--network=none",
            "--memory=8g",
            "--memory-swap=10g",
            "--cpus=4",
            "--entrypoint",
            "bash",
        ]
        for source, target, mode in (
            (root, "/workspace/codec-b043", "rw"),
            (WS / "helpers", "/workspace/helpers", "ro"),
            (WS / "source", "/workspace/source", "ro"),
            (WS / "cache", "/workspace/cache", "rw"),
            (Path("/opt/bc250-stack/current/torch"), "/workspace/pytorch-work", "ro"),
            (Path("/opt/bc250-stack/current/rocm"), "/workspace/build/dist/rocm", "ro"),
        ):
            command += ["-v", f"{source}:{target}:{mode}"]
        command += [
            "-e",
            "CODEC_WHEEL_DIR=/workspace/codec-b043/wheels-" + args.attempt,
            deps["image"],
            "/workspace/helpers/build_codec_b043.sh",
        ]
        run(command, "wheel", 3600)
        wheel = next((root / ("wheels-" + args.attempt)).glob("torchcodec-*.whl"))
        report.update(
            wheel=str(wheel),
            wheel_sha256=sha(wheel),
            source_archive_sha256=sha(root / "source.tar"),
            source_provenance=json.loads((root / "source-provenance.json").read_text()),
        )
    else:
        build = json.loads(
            (WS / ("logs/codec-b043-build-" + args.attempt + "/final.json")).read_text()
        )
        assert build["status"] == "passed"
        wheel = Path(build["wheel"])
        assert sha(wheel) == build["wheel_sha256"]
        context = root / ("runtime-context-" + args.attempt)
        (context / "wheels").mkdir(parents=True)
        shutil.copy2(wheel, context / "wheels" / wheel.name)
        tag = "localhost/bc250-stack:codec-b043-" + args.attempt
        run(
            [
                "podman",
                "build",
                "--pull=never",
                "--memory=8g",
                "--memory-swap=10g",
                "--tag",
                tag,
                "--file",
                str(WS / "helpers/Containerfile.codec-runtime-b043"),
                str(context),
            ],
            "image",
        )
        active = Path("/opt/bc250-stack/current").resolve(strict=True)
        candidate = root / ("candidate-release-" + args.attempt)
        candidate.mkdir()
        for entry in active.iterdir():
            if entry.name != "runtime.json":
                (candidate / entry.name).symlink_to(entry)
        manifest = json.loads((active / "runtime.json").read_text())
        previous_image = manifest["image"]
        manifest["image"] = image(tag)
        manifest["torchcodec_b043"] = dict(
            version="0.17.0+gfx1013",
            wheel_sha256=sha(wheel),
            source_revision=build["source_provenance"]["revision"],
            scope="CPU FFmpeg audio candidate; qualification pending; CUDA and image builds disabled.",
        )
        atomic_json(candidate / "runtime.json", manifest)
        backup = WS / "recovery/codec-b043-before"
        if not backup.exists():
            backup.mkdir()
            shutil.copy2(active / "runtime.json", backup / "runtime.json")
            shutil.copy2(
                WS / "helpers/qualify_training_n33_on_host.py",
                backup / "qualify_training_n33_on_host.py",
            )
        next_helper = WS / "helpers/qualify_training_n33.codec-next.py"
        assert next_helper.is_file()
        os.replace(next_helper, WS / "helpers/qualify_training_n33_on_host.py")
        report.update(
            image=manifest["image"],
            previous_image=previous_image,
            candidate_release=str(candidate),
            candidate_runtime_sha256=sha(candidate / "runtime.json"),
            wheel_sha256=sha(wheel),
        )
    report["status"] = "passed"
except BaseException as error:
    report.update(status="failed", error=repr(error))
    raise
finally:
    report["changed_protected_hashes"] = [
        n for n, h in protected.items() if sha(Path(n)) != h
    ]
    if report["changed_protected_hashes"]:
        report["status"] = "failed"
    atomic_json(logs / "final.json", report)
assert report["status"] == "passed"
