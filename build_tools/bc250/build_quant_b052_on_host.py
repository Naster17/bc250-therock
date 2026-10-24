#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Copy the qualified daily build and apply an exact source overlay under EX lock."""

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
    preflight("quant-b052-build")
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    baseline = json.loads(
        (WS / "logs/quant-b051-qualification-r1/final.json").read_text()
    )
    assert baseline["status"] == "passed"
    for name, digest in baseline["protected_sha256"].items():
        assert sha(Path(name)) == digest, name
    source = WS / "pytorch-distributed-b031"
    assert (
        source / "GIT_REVISION"
    ).read_text().strip() == "2b449c4abe36b896a6efb2a68e7a42d9922d5696"
    assert shutil.disk_usage(WS).free > 7 * 1024**3
    candidate = WS / "pytorch-quant-b052"
    candidate.mkdir()
    logs = WS / "logs/quant-b052-build"
    logs.mkdir()
    staged = WS / "staging/quant-b052-overlay"
    provenance = json.loads((staged / "source.json").read_text())
    for name, digest in provenance["source_sha256"].items():
        assert sha(staged / name) == digest, name
    report = dict(
        status="copying",
        pid=os.getpid(),
        container="bc250-quant-b052-build",
        candidate=str(candidate),
        source=provenance,
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
    for name in provenance["source_sha256"]:
        target = candidate / name
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.is_symlink() and target.resolve().is_relative_to(candidate)
        shutil.copy2(staged / name, target)
    (candidate / "GIT_REVISION").write_text(provenance["torch_revision"] + "\n")
    assert not os.path.samefile(
        source / "torch/lib/libtorch_hip.so", candidate / "torch/lib/libtorch_hip.so"
    )
    assert shutil.disk_usage(WS).free > 3 * 1024**3
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
    for src, dst, mode in (
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
    ):
        command += ["-v", f"{src}:{dst}:{mode}"]
    command += [image, "bash", "/workspace/helpers/build_torch_b052.sh"]
    report.update(status="building", image=image, command=command)
    atomic_json(logs / "progress.json", report)
    try:
        with (logs / "build.log").open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        atomic_json(logs / "exit.json", dict(returncode=result.returncode))
        assert result.returncode == 0
        for name, digest in baseline["protected_sha256"].items():
            assert sha(Path(name)) == digest, name
        for name, digest in provenance["source_sha256"].items():
            if name in (
                "torch/csrc/cuda/nccl.cpp",
                "torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp",
            ):
                reference = WS / "pytorch-rccl-b037" / name
                assert sha(reference) == baseline["protected_sha256"][str(reference)]
                assert sha(candidate / name) == sha(reference), name
            else:
                assert sha(candidate / name) == digest, name
        report["post_hipify_source_sha256"] = {
            name: sha(candidate / name) for name in provenance["source_sha256"]
        }
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
