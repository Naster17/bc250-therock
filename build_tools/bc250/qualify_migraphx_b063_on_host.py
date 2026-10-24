#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Qualify pinned header admission and preservation; no runtime deployment."""

import json
import os
from pathlib import Path
import subprocess

from bc250_torch import runtime_lock
from deploy_torch_a04_on_host import atomic_json, sha
from qualify_rccl_b038_on_host import gate
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight


def main():
    preflight("migraphx-b063-qualification")
    scratch = WS / "tmp/migraphx-b063/runtime-r2"
    scratch.mkdir(exist_ok=True)
    os.chown(scratch, 1000, 1000)
    baseline = json.loads(
        (WS / "logs/migraphx-b062-qualification-r1/final.json").read_text()
    )
    assert baseline["status"] == "passed"
    protected = dict(baseline["protected_sha256"])
    for path in (WS / "helpers").glob("*migraphx_b063*"):
        if path.is_file():
            protected[str(path)] = sha(path)
    protected[str(WS / "helpers/probe_migraphx_jit_b063.cpp")] = sha(
        WS / "helpers/probe_migraphx_jit_b063.cpp"
    )
    for path in (WS / "tmp/migraphx-b063/20261024-r4/tmp").glob("*"):
        if path.is_file():
            protected[str(path)] = sha(path)
    hashes = json.loads((WS / "migraphx-b063/source-sha256.json").read_text())
    for name, digest in hashes.items():
        assert sha(WS / "migraphx-b063/source" / name) == digest, name
    logs = WS / "logs/migraphx-b063-qualification-r2"
    logs.mkdir()
    before = subprocess.check_output(["dmesg"])
    (logs / "kernel-before.log").write_bytes(before)
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    manifest = json.loads(Path("/opt/bc250-stack/current/runtime.json").read_text())
    report = dict(
        status="running",
        gates=[],
        protected_sha256=protected,
        links=baseline["links"],
        aot_caches=baseline["aot_caches"],
        boot_id=boot,
        full_backend_built=False,
        runtime_changed=False,
        source_files_verified=len(hashes),
    )
    try:
        with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
            for name, digest in protected.items():
                assert sha(Path(name)) == digest, name
            for name, launcher, script, args, count in (
                ("native", "bc250-torch", "validate_migraphx_b063.py", [], 71),
                ("core", "bc250-train", "validate_cleanup_runtime.py", [], 7),
                (
                    "onnx",
                    "bc250-torch",
                    "validate_onnx_b061.py",
                    ["--group", "audit"],
                    4,
                ),
                (
                    "daily",
                    "bc250-train",
                    "validate_rccl_b038.py",
                    ["--daily", "--consumers"],
                    5,
                ),
                ("rccl", "bc250-train", "validate_rccl_b038.py", ["--consumers"], 6),
            ):
                command = ["/usr/local/bin/" + launcher]
                if name == "rccl":
                    command += ["--runtime-profile", "rccl-single"]
                command += [str(WS / "helpers" / script), *args]
                report["gates"].append(
                    gate("migraphx-b063-r2-" + name, command, count, manifest)
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
            n for n, d in protected.items() if sha(Path(n)) != d
        ]
        report["links_unchanged"] = all(
            os.readlink(p) == v for p, v in baseline["links"].items()
        )
        report["kernel_log_unchanged"] = before == after
        report["running_containers"] = subprocess.check_output(
            ["podman", "ps", "-q"], text=True
        ).split()
        if (
            report["changed_hashes"]
            or not report["links_unchanged"]
            or before != after
            or report["running_containers"]
        ):
            report["status"] = "failed"
        atomic_json(logs / "final.json", report)
    assert (
        report["status"] == "passed" and sum(g["checks"] for g in report["gates"]) == 93
    )


if __name__ == "__main__":
    main()
