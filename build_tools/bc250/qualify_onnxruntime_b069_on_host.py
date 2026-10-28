#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Guard native async SDK admission and retain Torch/CPU ORT consumers unchanged."""
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
    assert args.attempt in ("r1", "r2", "r3")
    preflight("onnxruntime-b069-qualification-" + args.attempt)
    logs = WS / ("logs/onnxruntime-b069-qualification-" + args.attempt)
    logs.mkdir()
    built = WS / "tmp/onnxruntime-b069/probe-r4"
    build = json.loads((built / "final.json").read_text())
    assert build["status"] == "built" and build["c_api_symbols_linked"] == 168
    baseline = json.loads(
        (WS / "logs/migraphx-b068-qualification-r1/final.json").read_text()
    )
    protected = dict(baseline["protected_sha256"])
    for path in [
        *(WS / "helpers").glob("*onnxruntime*b069*"),
        WS / "helpers/migraphx-b069-cpp20.hpp",
        WS / "helpers/migraphx-b069-cpp20-test.cpp",
        built / "build/admission_probe",
        built / "final.json",
    ]:
        if path.is_file():
            protected[str(path)] = sha(path)
    for name, digest in build["source_sha256"].items():
        protected[str(built / name)] = digest
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
        active_runtime_changed=False,
        host105_contacted=False,
        ort_provider_built=False,
    )
    try:
        with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
            assert not verify_hashes(protected)
            registered(LaunchOptions())
            registered(LaunchOptions(runtime_profile="rccl-single"))
            for profile in ("daily", "rccl-single"):
                label = "onnxruntime-b069-" + args.attempt + "-probe-" + profile
                scratch = WS / "tmp/onnxruntime-b069" / label
                scratch.mkdir()
                os.chown(scratch, manifest["uid"], manifest["gid"])
                command = [
                    "/usr/local/bin/bc250-migraphx",
                    "--runtime-profile",
                    profile,
                ]
                for name, value in dict(
                    TMPDIR=str(scratch),
                    TMP=str(scratch),
                    TEMP=str(scratch),
                    MIOPEN_USER_DB_PATH=str(scratch / "miopen-db"),
                    MIOPEN_CUSTOM_CACHE_DIR=str(scratch / "miopen-cache"),
                ).items():
                    command += ["--env", name + "=" + value]
                command += [
                    str(WS / "helpers/validate_onnxruntime_b069.py"),
                    "--probe",
                    str(built / "build/admission_probe"),
                ]
                report["gates"].append(gate(label, command, 8, manifest))
                atomic_json(logs / "progress.json", report)
                print("PASS", label, 8, flush=True)
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
                        "onnxruntime-b069-" + args.attempt + "-" + name,
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
        report["status"] == "passed" and sum(g["checks"] for g in report["gates"]) == 38
    )
    print("ONNXRUNTIME_B069_ADMISSION_QUALIFIED", flush=True)


if __name__ == "__main__":
    main()
