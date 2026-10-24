#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Preserve daily/opt-in payloads and qualify the isolated W8 recipe on .104."""

import json
import os
from pathlib import Path
import subprocess

from bc250_torch import runtime_lock
from deploy_torch_a04_on_host import atomic_json, sha
from qualify_rccl_b038_on_host import gate
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight

SOURCES = (
    "audit_quant_b051.py",
    "bc250_quant_b051.py",
    "validate_quant_b051.py",
    "qualify_quant_b051_on_host.py",
)


def accepted(label, count):
    root = WS / "logs" / label
    assert json.loads((root / "exit.json").read_text()) == dict(
        returncode=0, kernel_log_unchanged=True, running_containers=[]
    )
    result = json.loads((root / "result.json").read_text())
    assert result["status"] == "passed" and len(result["checks"]) == count
    guard = json.loads((Path("/run") / label / "state.json").read_text())
    assert guard["status"] == "acknowledged" and guard["timeout_seconds"] == 600
    return dict(label=label, checks=count, guard=guard)


def main():
    preflight("quant-b051-qualification")
    logs = WS / "logs/quant-b051-qualification-r1"
    logs.mkdir()
    baseline = json.loads(
        (WS / "logs/rccl-b038-qualification-r1/final.json").read_text()
    )
    assert baseline["status"] == "passed"
    protected = dict(baseline["protected_sha256"])
    for name in SOURCES:
        protected[str(WS / "helpers" / name)] = sha(WS / "helpers" / name)
    protected[str(WS / "helpers/update_dev_prefix.py")] = sha(
        WS / "helpers/update_dev_prefix.py"
    )
    source = Path("/opt/bc250-stack/current/torch")
    provenance = {
        name: sha(source / name)
        for name in (
            "GIT_REVISION",
            "aten/src/ATen/native/cuda/int8mm.cu",
            "aten/src/ATen/native/cuda/int4mm.cu",
            "aten/src/ATen/native/native_functions.yaml",
            "tools/autograd/derivatives.yaml",
        )
    }
    assert (
        provenance["aten/src/ATen/native/cuda/int8mm.cu"]
        == "7f084841d830ac257507ea0d2d1df7aefc57ff3e4e84113177f1e23b6632d846"
    )
    assert (
        provenance["aten/src/ATen/native/cuda/int4mm.cu"]
        == "02dde40cf3cee27a2c3ab2c202b13febb8428074a928ba9c1a577317a8b2d020"
    )
    report = dict(
        status="running",
        gates=[],
        protected_sha256=protected,
        links=baseline["links"],
        aot_caches=baseline["aot_caches"],
        source_provenance_sha256=provenance,
        native_torch_revision=(source / "GIT_REVISION").read_text().strip(),
        native_build_required=False,
        default_changed=False,
        host105_contacted=False,
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    )
    before = subprocess.check_output(["dmesg"])
    (logs / "kernel-before.log").write_bytes(before)
    daily = Path("/opt/bc250-stack/current").resolve(strict=True)
    manifest = json.loads((daily / "runtime.json").read_text())
    try:
        with runtime_lock(daily / ".runtime.lock"):
            for name, digest in protected.items():
                assert sha(Path(name)) == digest, name
            for label, count in [
                ("quant-b051-audit-r2", 7),
                ("quant-b051-prototype-r1", 71),
            ]:
                report["gates"].append(accepted(label, count))
            for name, script, arguments, count in (
                ("core", "validate_cleanup_runtime.py", [], 7),
                (
                    "compiled-dense",
                    "validate_compiler_n310.py",
                    ["--family", "dense"],
                    14,
                ),
                ("daily", "validate_rccl_b038.py", ["--daily", "--consumers"], 5),
                ("opt-in", "validate_rccl_b038.py", ["--consumers"], 6),
            ):
                command = ["/usr/local/bin/bc250-train"]
                if name == "opt-in":
                    command += ["--runtime-profile", "rccl-single"]
                command += [str(WS / "helpers" / script), *arguments]
                report["gates"].append(
                    gate("quant-b051-r1-" + name, command, count, manifest)
                )
                atomic_json(logs / "progress.json", report)
                print("PASS", name, count, flush=True)
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
        report["links_unchanged"] = all(
            os.readlink(p) == v for p, v in baseline["links"].items()
        )
        report["kernel_log_unchanged_entire_batch"] = after == before
        report["boot_unchanged"] = (
            Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            == report["boot_id"]
        )
        report["running_containers"] = subprocess.check_output(
            ["podman", "ps", "-q"], text=True
        ).split()
        report["disk_available_bytes"] = (
            os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize
        )
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
    print("QUANT_B051_PASS", flush=True)


if __name__ == "__main__":
    main()
