#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export qualified native layout-fix payload, logs, source and recovery evidence."""

import json
import os
from pathlib import Path
import subprocess
import tarfile

from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS

SOURCES = (
    "build_quant_b052_on_host.py",
    "build_torch_b052.sh",
    "seal_quant_b052_build.py",
    "qualify_quant_b052_on_host.py",
    "validate_quant_b052.py",
    "deploy_quant_b052_on_host.py",
    "qualify_quant_b052_ordinary.py",
    "export_quant_b052_on_host.py",
)


def main():
    assert os.geteuid() == 0
    native = json.loads(
        (WS / "logs/quant-b052-qualification-r1/final.json").read_text()
    )
    ordinary = json.loads((WS / "logs/quant-b052-ordinary.json").read_text())
    built = json.loads((WS / "logs/quant-b052-build/final.json").read_text())
    assert native["status"] == ordinary["status"] == "passed"
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    for name, digest in ordinary["protected_sha256"].items():
        assert sha(Path(name)) == digest, name
    first = WS / "logs/quant-b052-r1-layouts/kernel-before.log"
    assert subprocess.check_output(["dmesg"]) == first.read_bytes()
    guards = [
        dict(label=g["label"], status=g["guard"]["status"])
        for g in [*native["gates"], *ordinary["gates"]]
    ]
    assert len(guards) == 14 and all(g["status"] == "acknowledged" for g in guards)
    candidate = Path(built["candidate"])
    destination = WS / "quant-b052/export"
    destination.mkdir()
    report = dict(
        status="passed",
        candidate_checks=sum(g["checks"] for g in native["gates"]),
        ordinary_checks=sum(g["checks"] for g in ordinary["gates"]),
        guards=guards,
        protected_paths=len(ordinary["protected_sha256"]),
        links=ordinary["links"],
        aot_caches=ordinary["aot_caches"],
        runtime_sha256=ordinary["runtime_sha256"],
        old_runtime_sha256=sha(WS / "recovery/quant-b052-before/runtime.json"),
        opt_in_runtime_sha256=sha(
            Path("/opt/bc250-stack/profiles/rccl-single/runtime.json")
        ),
        torch_revision=built["source"]["torch_revision"],
        torch_target=ordinary["torch_target"],
        native_sha256=built["native_sha256"],
        generated_hip_sha256=built["generated_hip_sha256"],
        post_hipify_source_sha256=built["post_hipify_source_sha256"],
        source=built["source"],
        source_sha256={n: sha(WS / "helpers" / n) for n in SOURCES},
        build_image=built["image"],
        build_returncode=0,
        build_container=built["container"],
        kernel=os.uname().release,
        boot_id=native["boot_id"],
        kernel_log_unchanged_including_ordinary=True,
        running_jobs=[],
        host105_contacted=False,
        native_backward="explicitly unsupported",
        int4="CDNA2 gate retained",
        opt_in_rccl="unchanged old build; native layout repair not yet applied",
        rollback="identity-guarded helper retained; not exercised",
        disk_available_bytes=os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize,
    )
    assert report["candidate_checks"] == 947 and report["ordinary_checks"] == 231
    atomic_json(destination / "verification.json", report)
    archive = destination / "evidence.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for p in sorted((WS / "logs").glob("quant-b052-*")):
            tar.add(p, arcname="logs/" + p.name)
        for n in SOURCES:
            tar.add(WS / "helpers" / n, arcname="helpers/" + n)
        tar.add(WS / "staging/quant-b052-overlay", arcname="source-overlay")
        tar.add(WS / "recovery/quant-b052-before", arcname="recovery")
        for g in guards:
            tar.add(Path("/run") / g["label"], arcname="guards/" + g["label"])
        tar.add(candidate / "torch/lib", arcname="native/torch/lib")
        for p in (candidate / "torch").glob("_C*.so"):
            tar.add(p, arcname="native/torch/" + p.name)
        for n in (
            "GIT_REVISION",
            "torch/version.py",
            "build/CMakeCache.txt",
            "aten/src/ATen/native/hip/int8mm.hip",
        ):
            tar.add(candidate / n, arcname="native/" + n)
        for n in built["post_hipify_source_sha256"]:
            tar.add(candidate / n, arcname="post-hipify-source/" + n)
        tar.add(Path("/opt/bc250-stack/current/runtime.json"), arcname="runtime.json")
        tar.add(destination / "verification.json", arcname="verification.json")
    atomic_json(
        destination / "archive.json",
        dict(sha256=sha(archive), bytes=archive.stat().st_size),
    )
    for p in destination.iterdir():
        os.chown(p, 1000, 1000)
    print("QUANT_B052_EXPORT_PASS", flush=True)


if __name__ == "__main__":
    main()
