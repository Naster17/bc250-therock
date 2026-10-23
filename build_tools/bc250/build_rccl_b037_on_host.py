#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a RCCL-enabled independent qualified Torch candidate, never active source."""

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
    preflight("rccl-b037-build")
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    baseline = json.loads(
        (WS / "logs/rccl-b036-qualification-r1/final.json").read_text()
    )
    for path, expected in baseline["protected_sha256"].items():
        assert sha(Path(path)) == expected, path
    assert shutil.disk_usage(WS).free > 10 * 1024**3
    source = WS / "pytorch-rccl-b035"
    assert (
        source / "GIT_REVISION"
    ).read_text().strip() == "ade6fcc1ae7574b2f5ad285c8a2c592af1e3e99c"
    candidate = WS / "pytorch-rccl-b037"
    candidate.mkdir()
    staged = WS / "staging/rccl-b037"
    staged.mkdir()
    shutil.copyfile(WS / "helpers/b037-source.json", staged / "source.json")
    provenance = json.loads((staged / "source.json").read_text())
    assert provenance["base_revision"] == "ade6fcc1ae7574b2f5ad285c8a2c592af1e3e99c"
    for name, expected in provenance["source_sha256"].items():
        assert sha(WS / "helpers" / ("b037-" + Path(name).name)) == expected, name
    logs = WS / "logs/rccl-b037-build"
    logs.mkdir()
    report = dict(
        status="copying",
        pid=os.getpid(),
        container="bc250-rccl-b037-build",
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
    for name in provenance["source_sha256"]:
        shutil.copyfile(WS / "helpers" / ("b037-" + Path(name).name), candidate / name)
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
    command += [image, "bash", "/workspace/helpers/build_torch_b037.sh"]
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
