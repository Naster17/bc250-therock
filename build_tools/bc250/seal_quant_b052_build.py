#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Seal an exit-zero pip build after auditing automatic HIP source generation."""

import json
import shutil
from pathlib import Path

from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS


def main():
    logs = WS / "logs/quant-b052-build"
    report = json.loads((logs / "final.json").read_text())
    assert (
        report["status"] == "failed"
        and report["error"] == "AssertionError('torch/csrc/cuda/nccl.cpp')"
    )
    assert json.loads((logs / "exit.json").read_text())["returncode"] == 0
    baseline = json.loads(
        (WS / "logs/quant-b051-qualification-r1/final.json").read_text()
    )
    for name, digest in baseline["protected_sha256"].items():
        assert sha(Path(name)) == digest, name
    candidate = Path(report["candidate"])
    for name, digest in report["source"]["source_sha256"].items():
        if name in (
            "torch/csrc/cuda/nccl.cpp",
            "torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp",
        ):
            reference = WS / "pytorch-rccl-b037" / name
            assert sha(reference) == baseline["protected_sha256"][str(reference)]
            assert sha(candidate / name) == sha(reference), name
        else:
            assert sha(candidate / name) == digest, name
    original = (
        WS / "pytorch-distributed-b031/aten/src/ATen/native/hip/int8mm.hip"
    ).read_bytes()
    expected = original.replace(
        b"x.to(at::kFloat);", b"x.to(at::kFloat).contiguous();"
    ).replace(b"scale.to(at::kFloat);", b"scale.to(at::kFloat).contiguous();")
    generated = candidate / "aten/src/ATen/native/hip/int8mm.hip"
    assert generated.read_bytes() == expected
    cmake = (candidate / "build/CMakeCache.txt").read_text()
    flags = dict(
        line.split("=", 1)
        for line in cmake.splitlines()
        if "=" in line and not line.startswith(("#", "//"))
    )
    assert flags["USE_NCCL:BOOL"] in ("0", "OFF")
    assert flags["USE_GLOO:BOOL"] in ("1", "ON")
    failure = logs / "postcheck-failure.json"
    assert not failure.exists()
    shutil.copyfile(logs / "final.json", failure)
    report.pop("error")
    report.update(
        status="built-not-qualified",
        postcheck_resolution="Pip exit0; two auto-HIP-translated NCCL source files exactly match protected B03.7 sources; NCCL off. Generated int8mm HIP delta exactly two contiguous calls.",
        post_hipify_source_sha256={
            n: sha(candidate / n) for n in report["source"]["source_sha256"]
        },
        generated_hip_sha256=sha(generated),
        native_sha256={p.name: sha(p) for p in (candidate / "torch/lib").glob("*.so")},
    )
    atomic_json(logs / "final.json", report)
    print("QUANT_B052_BUILD_SEALED", flush=True)


if __name__ == "__main__":
    main()
