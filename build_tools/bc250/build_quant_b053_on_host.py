#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Copy the qualified RCCL build and apply an exact source overlay under EX lock."""

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
    preflight("quant-b053-build")
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    baseline = json.loads((WS / "logs/quant-b052-ordinary.json").read_text())
    assert baseline["status"] == "passed"
    for name, digest in baseline["protected_sha256"].items():
        assert sha(Path(name)) == digest, name
    source = WS / "pytorch-rccl-b037"
    assert (
        source / "GIT_REVISION"
    ).read_text().strip() == "14cfef2d8c5e8c2c40e3b24587233227ea8dadc3"
    recovery = WS / "recovery/quant-b053-before"
    recovery.mkdir()
    for name in ("validate_quant_b052.py", "validate_torch_a04_on_host.py"):
        original = WS / "helpers" / name
        shutil.copy2(original, recovery / name)
        original_digest = sha(original)
        shutil.copy2(WS / "helpers/b053-next" / name, original)
        baseline["protected_sha256"][str(original)] = sha(original)
        baseline["protected_sha256"][str(recovery / name)] = original_digest
    baseline_dir = WS / "logs/quant-b053-baseline"
    baseline_dir.mkdir()
    atomic_json(baseline_dir / "final.json", baseline)
    source_bytes = (
        int(subprocess.check_output(["du", "-sk", str(source)], text=True).split()[0])
        * 1024
    )
    assert shutil.disk_usage(WS).free > source_bytes + 5 * 1024**3 // 4
    candidate = WS / "pytorch-quant-b053"
    candidate.mkdir()
    logs = WS / "logs/quant-b053-build"
    logs.mkdir()
    staged = WS / "staging/quant-b053-overlay"
    provenance = json.loads((staged / "source.json").read_text())
    for name, digest in provenance["source_sha256"].items():
        assert sha(staged / name) == digest, name
    report = dict(
        status="copying",
        pid=os.getpid(),
        container="bc250-quant-b053-build",
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
    assert shutil.disk_usage(WS).free > 5 * 1024**3 // 4
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
        (WS / "rccl-b034/install-r1", str(WS / "rccl-b034/install-r1"), "ro"),
    ):
        command += ["-v", f"{src}:{dst}:{mode}"]
    command += [image, "bash", "/workspace/helpers/build_torch_b053.sh"]
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
            assert sha(candidate / name) == digest, name
        generated = candidate / "aten/src/ATen/native/hip/int8mm.hip"
        original = (source / "aten/src/ATen/native/hip/int8mm.hip").read_bytes()
        expected = original.replace(
            b"x.to(at::kFloat);", b"x.to(at::kFloat).contiguous();"
        ).replace(b"scale.to(at::kFloat);", b"scale.to(at::kFloat).contiguous();")
        assert generated.read_bytes() == expected
        flags = dict(
            line.split("=", 1)
            for line in (candidate / "build/CMakeCache.txt").read_text().splitlines()
            if "=" in line and not line.startswith(("#", "//"))
        )
        assert flags["USE_NCCL:BOOL"] in ("1", "ON")
        assert flags["USE_SYSTEM_NCCL:BOOL"] in ("1", "ON")
        assert flags["USE_GLOO:BOOL"] in ("1", "ON")
        report["generated_hip_sha256"] = sha(generated)
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
