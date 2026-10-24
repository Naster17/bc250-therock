#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Qualify a separate matched RCCL profile under EX before registry activation."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from bc250_torch import RCCL_SINGLE_ENVIRONMENT
from deploy_torch_a04_on_host import atomic_json, sha, check_gate
from qualify_rccl_b037_on_host import GATES as RETAINED
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight

PROFILE = Path("/opt/bc250-stack/dev/rccl-single-b053")
GATES = (
    ("layouts", "validate_quant_b052.py", ["--nccl-available"], 220),
    ("recipe", "validate_quant_b051.py", [], 71),
) + RETAINED


def main():
    preflight("quant-b053-qualification")
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    baseline = json.loads((WS / "logs/quant-b053-baseline/final.json").read_text())
    built = json.loads((WS / "logs/quant-b053-build/final.json").read_text())
    assert baseline["status"] == "passed" and built["status"] == "built-not-qualified"
    protected = dict(baseline["protected_sha256"])
    candidate = Path(built["candidate"])
    for name, digest in built["native_sha256"].items():
        protected[str(candidate / "torch/lib" / name)] = digest
    for name, digest in built["post_hipify_source_sha256"].items():
        protected[str(candidate / name)] = digest
    for name in (
        "GIT_REVISION",
        "build/CMakeCache.txt",
        "aten/src/ATen/native/hip/int8mm.hip",
    ):
        protected[str(candidate / name)] = sha(candidate / name)
    for name in (
        "build_quant_b053_on_host.py",
        "build_torch_b053.sh",
        "qualify_quant_b053_on_host.py",
    ):
        protected[str(WS / "helpers" / name)] = sha(WS / "helpers" / name)
    for name, digest in protected.items():
        assert sha(Path(name)) == digest, name
    registry = Path("/opt/bc250-stack/profiles/rccl-single")
    old = registry.resolve(strict=True)
    assert (
        sha(old / "runtime.json")
        == "42eb7631d6b760b0fabed37c35ac57c9e02dbc2a7ae44780efd63c5e61a815b9"
    )
    daily = Path("/opt/bc250-stack/current").resolve(strict=True)
    PROFILE.mkdir()
    for name in ("rocm", "cache", "python-consumers", "rocblas-gemm-n32.csv"):
        (PROFILE / name).symlink_to(daily / name)
    (PROFILE / "torch").symlink_to(candidate)
    shutil.copytree(old / "rccl", PROFILE / "rccl", symlinks=True)
    identity = {
        "rccl/lib/librccl.so.1.0": sha(PROFILE / "rccl/lib/librccl.so.1.0"),
        "torch/GIT_REVISION": sha(candidate / "GIT_REVISION"),
    }
    assert (
        identity["rccl/lib/librccl.so.1.0"]
        == "ca790dc73b76a94a7dc97538608cd94b789e73cc18d756b0631111c18230583e"
    )
    for name, digest in built["native_sha256"].items():
        identity["torch/torch/lib/" + name] = digest
    manifest = json.loads((old / "runtime.json").read_text())
    manifest.update(
        release="dev-rccl-single-b053",
        torch_revision=built["source"]["torch_revision"],
        qualification="Bounded B05.3 outcome in rccl_single_profile.packaged_evidence",
    )
    manifest["rccl_single_profile"].update(
        identity_sha256=identity,
        environment=RCCL_SINGLE_ENVIRONMENT,
        native_evidence=str(WS / "logs/quant-b053-qualification-r1/final.json"),
        packaged_evidence=str(WS / "logs/quant-b053-ordinary.json"),
    )
    manifest["quant_b053"] = dict(
        source=built["source"],
        build_evidence=str(WS / "logs/quant-b053-build/final.json"),
        scope="Native layout correction with retained one-device RCCL contracts; backward/INT4 and other recorded exclusions unchanged",
    )
    manifest["source_state"] += (
        "; B05.3 matched RCCL native weight-only INT8 layout repair"
    )
    manifest["rccl_b035"].update(
        status="B05.3 native layout correction; bounded RCCL subset retained",
        build_evidence=str(WS / "logs/quant-b053-build/final.json"),
        source=built["source"],
        rccl=str(PROFILE / "rccl"),
    )
    atomic_json(PROFILE / "runtime.json", manifest)
    protected[str(PROFILE / "runtime.json")] = sha(PROFILE / "runtime.json")
    for name, digest in identity.items():
        protected[str(PROFILE / name)] = digest
    links = dict(baseline["links"])
    links[str(registry)] = os.readlink(registry)
    logs = WS / "logs/quant-b053-qualification-r1"
    logs.mkdir()
    before = subprocess.check_output(["dmesg"])
    (logs / "kernel-before.log").write_bytes(before)
    report = dict(
        status="running",
        gates=[],
        protected_sha256=protected,
        links=links,
        aot_caches=baseline["aot_caches"],
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        profile=str(PROFILE),
        profile_sha256=sha(PROFILE / "runtime.json"),
        old_profile=str(old),
        source=built["source"],
        native_sha256=built["native_sha256"],
    )
    atomic_json(logs / "before.json", report)
    try:
        for group, script, arguments, count in GATES:
            arguments = list(arguments)
            if script in (
                "validate_rccl_b035.py",
                "validate_rccl_b036.py",
                "validate_rccl_b037.py",
            ):
                arguments += ["--packaged"]
            label = "quant-b053-r1-" + group
            command = [
                sys.executable,
                str(WS / "helpers/validate_torch_a04_on_host.py"),
                "--guard-release",
                os.uname().release,
                "--release",
                str(PROFILE),
                "--runtime-profile",
                "rccl-single",
                "--label",
                label,
                "--script",
                script,
                *["--test-arg=" + a for a in arguments],
            ]
            with (logs / (group + ".log")).open("w") as stream:
                subprocess.run(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=570,
                )
            check_gate(label)
            result = json.loads((WS / "logs" / label / "result.json").read_text())
            assert result["status"] == "passed" and len(result["checks"]) == count
            guard = json.loads((Path("/run") / label / "state.json").read_text())
            assert guard["status"] == "acknowledged"
            report["gates"].append(dict(label=label, checks=count, guard=guard))
            atomic_json(logs / "progress.json", report)
            print("PASS", group, count, flush=True)
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        after = subprocess.check_output(["dmesg"])
        (logs / "kernel-after.log").write_bytes(after)
        report["changed_hashes"] = [
            p for p, d in protected.items() if sha(Path(p)) != d
        ]
        report["links_unchanged"] = all(os.readlink(p) == v for p, v in links.items())
        report["kernel_log_unchanged_entire_batch"] = before == after
        report["boot_unchanged"] = (
            Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            == report["boot_id"]
        )
        report["running_containers"] = subprocess.check_output(
            ["podman", "ps", "-q"], text=True
        ).split()
        if (
            report["changed_hashes"]
            or not report["links_unchanged"]
            or not report["kernel_log_unchanged_entire_batch"]
            or not report["boot_unchanged"]
            or report["running_containers"]
        ):
            report["status"] = "failed"
        atomic_json(logs / "final.json", report)
    assert report["status"] == "passed"
    print("QUANT_B053_PASS", flush=True)


if __name__ == "__main__":
    main()
