#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a RCCL-enabled independent qualified Torch candidate, never active source."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight


def main():
    assert os.geteuid() == 0
    preflight("rccl-b035-build")
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    baseline = json.loads(
        (WS / "logs/rccl-b034-qualification-r1/final.json").read_text()
    )
    for path, expected in baseline["protected_sha256"].items():
        assert sha(Path(path)) == expected, path
    assert shutil.disk_usage(WS).free > 10 * 1024**3
    source = (Path("/opt/bc250-stack/current") / "torch").resolve()
    assert source == WS / "pytorch-distributed-b031"
    candidate = WS / "pytorch-rccl-b035"
    candidate.mkdir()
    staged = WS / "staging/rccl-b035"
    provenance = json.loads((staged / "source.json").read_text())
    assert provenance["torch_revision"] == "2b449c4abe36b896a6efb2a68e7a42d9922d5696"
    assert (
        sha(WS / "rccl-b034/install-r1/lib/librccl.so")
        == "ca790dc73b76a94a7dc97538608cd94b789e73cc18d756b0631111c18230583e"
    )
    assert sha(staged / "torch-source.tar") == provenance["source_archive_sha256"]
    logs = WS / "logs/rccl-b035-build"
    logs.mkdir()
    report = dict(
        status="copying",
        pid=os.getpid(),
        container="bc250-rccl-b035-build",
        source=provenance,
        candidate=str(candidate),
        protected_sha256=baseline["protected_sha256"],
        active_prefix_changed=False,
    )
    atomic_json(logs / "progress.json", report)
    writer = subprocess.Popen(
        ["tar", "-C", str(source), "--exclude=./third_party", "-cf", "-", "."],
        stdout=subprocess.PIPE,
    )
    subprocess.run(
        ["tar", "-C", str(candidate), "-xf", "-"], stdin=writer.stdout, check=True
    )
    writer.stdout.close()
    assert writer.wait() == 0
    with tarfile.open(staged / "torch-source.tar") as archive:
        archive.extractall(candidate, filter="data")
    (candidate / "GIT_REVISION").write_text(provenance["torch_revision"] + "\n")
    assert shutil.disk_usage(WS).free > 7 * 1024**3
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
    command = [
        "podman",
        "run",
        "--name",
        report["container"],
        "--network=none",
        "--cpus=4",
        "--memory=10g",
        "--memory-swap=12g",
    ]
    for src, dst, mode in [
        (WS / "source", "/workspace/source", "ro"),
        (WS / "helpers", "/workspace/helpers", "ro"),
        (candidate, "/workspace/pytorch-work", "rw"),
        (
            WS / "pytorch-sparse-followup/third_party",
            "/workspace/pytorch-work/third_party",
            "ro",
        ),
        (WS / "build", "/workspace/build", "ro"),
        (WS / "cache", "/workspace/cache", "rw"),
        (logs, "/workspace/logs", "rw"),
        (WS / "rccl-b034/install-r1", str(WS / "rccl-b034/install-r1"), "ro"),
    ]:
        command += ["-v", f"{src}:{dst}:{mode}"]
    command += [image, "bash", "/workspace/helpers/build_torch_b035.sh"]
    report.update(status="building", image=image, command=command)
    atomic_json(logs / "progress.json", report)
    atomic_json(logs / "command.json", command)
    try:
        with (logs / "build.log").open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        atomic_json(logs / "exit.json", dict(returncode=result.returncode))
        assert result.returncode == 0, result.returncode
        for path, expected in baseline["protected_sha256"].items():
            assert sha(Path(path)) == expected, path
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
