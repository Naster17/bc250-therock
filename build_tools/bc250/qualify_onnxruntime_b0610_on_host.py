#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Qualify the isolated matching ORT provider with real assignment and retained consumers."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from bc250_migraphx import registered
from bc250_torch import LaunchOptions, runtime_lock
from build_migraphx_b067_on_host import verify_hashes
from deploy_torch_a04_on_host import atomic_json, sha
from qualify_rccl_b038_on_host import gate
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True)
    args = parser.parse_args()
    assert args.attempt in ("r1", "r2", "r3", "r4")
    preflight("onnxruntime-b0610-qualification-" + args.attempt)
    base = WS / "tmp/onnxruntime-b0610"
    build = json.loads((base / "build-r2/final.json").read_text())
    assert build["status"] == "passed"
    candidate = base / "task-r2/build"
    assert (candidate / "onnxruntime/__init__.py").exists()
    baseline = json.loads(
        (WS / "logs/onnxruntime-b069-qualification-r2/final.json").read_text()
    )
    protected = dict(baseline["protected_sha256"])
    for path in (candidate / "onnxruntime").rglob("*"):
        if path.is_file():
            protected[str(path)] = sha(path)
    for path in (WS / "helpers").glob("*onnxruntime*b0610*"):
        if path.is_file():
            protected[str(path)] = sha(path)
    logs = WS / ("logs/onnxruntime-b0610-qualification-" + args.attempt)
    logs.mkdir()
    before = subprocess.check_output(["dmesg"])
    (logs / "kernel-before.log").write_bytes(before)
    manifest = json.loads(Path("/opt/bc250-stack/current/runtime.json").read_text())
    report = dict(
        status="running",
        gates=[],
        protected_sha256=protected,
        links=baseline["links"],
        aot_caches=baseline["aot_caches"],
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        runtime_changed=False,
        ort_candidate=str(candidate),
        host105_contacted=False,
    )
    try:
        with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
            assert not verify_hashes(protected)
            registered(LaunchOptions())
            registered(LaunchOptions(runtime_profile="rccl-single"))
            for profile, order in [
                ("daily", "ort"),
                ("daily", "torch"),
                ("rccl-single", "migraphx"),
            ]:
                label = (
                    "onnxruntime-b0610-" + args.attempt + "-" + profile + "-" + order
                )
                scratch = base / label
                scratch.mkdir()
                os.chown(scratch, manifest["uid"], manifest["gid"])
                command = [
                    "/usr/local/bin/bc250-migraphx",
                    "--runtime-profile",
                    profile,
                ]
                environment = dict(
                    TMPDIR=str(scratch),
                    TMP=str(scratch),
                    TEMP=str(scratch),
                    MIOPEN_USER_DB_PATH=str(scratch / "miopen-db"),
                    MIOPEN_CUSTOM_CACHE_DIR=str(scratch / "miopen-cache"),
                )
                for flag in ("FP16", "BF16", "FP8", "INT8", "EXHAUSTIVE_TUNE"):
                    environment[
                        "ORT_MIGRAPHX_"
                        + (flag + "_ENABLE" if flag != "EXHAUSTIVE_TUNE" else flag)
                    ] = "0"
                for name, value in environment.items():
                    command += ["--env", name + "=" + value]
                command += [
                    str(WS / "helpers/validate_onnxruntime_b0610.py"),
                    "--candidate",
                    str(candidate),
                    "--order",
                    order,
                ]
                report["gates"].append(gate(label, command, 32, manifest))
                atomic_json(logs / "progress.json", report)
                print("PASS", label, 32, flush=True)
            for name, launcher, script, extra, count in [
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
            ]:
                command = ["/usr/local/bin/" + launcher]
                if name == "rccl":
                    command += ["--runtime-profile", "rccl-single"]
                command += [str(WS / "helpers" / script), *extra]
                report["gates"].append(
                    gate(
                        "onnxruntime-b0610-" + args.attempt + "-" + name,
                        command,
                        count,
                        manifest,
                    )
                )
                atomic_json(logs / "progress.json", report)
            report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        after = subprocess.check_output(["dmesg"])
        (logs / "kernel-after.log").write_bytes(after)
        report["changed_hashes"] = verify_hashes(protected)
        report["links_unchanged"] = all(
            os.readlink(p) == v for p, v in report["links"].items()
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
        report["status"] == "passed"
        and sum(g["checks"] for g in report["gates"]) == 118
    )
    print("ORT_B0610_CANDIDATE_QUALIFIED", flush=True)


if __name__ == "__main__":
    main()
