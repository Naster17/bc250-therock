#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded isolated CK factory admission build and guarded current-profile checks."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from bc250_torch import runtime_lock
from build_migraphx_b067_on_host import verify_hashes
from deploy_torch_a04_on_host import atomic_json, sha
from qualify_rccl_b038_on_host import gate
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attempt", choices=("r1", "r2", "r3"), required=True)
    args = p.parse_args()
    preflight("ck-plugin-b071-" + args.attempt)
    assert os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize > 4 * 1024**3
    base = WS / "tmp/ck-plugin-b071"
    base.mkdir(exist_ok=True)
    output = base / ("build-" + args.attempt)
    output.mkdir()
    logs = WS / ("logs/ck-plugin-b071-qualification-" + args.attempt)
    logs.mkdir()
    a = json.loads(
        (WS / "logs/onnxruntime-b0612-qualification-r1/final.json").read_text()
    )
    protected = dict(a["protected_sha256"])
    stage = WS / "build/math-libs/composable_kernel/stage"
    for name in (
        "lib/libdevice_conv_operations.a",
        "include/ck/config.h",
        "include/ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp",
    ):
        path = stage / name
        protected[str(path)] = sha(path)
    for path in (WS / "helpers").glob("*b071*"):
        if path.is_file():
            protected[str(path)] = sha(path)
    before = subprocess.check_output(["dmesg"])
    (logs / "kernel-before.log").write_bytes(before)
    runtime = json.loads(Path("/opt/bc250-stack/current/runtime.json").read_text())
    r = dict(
        status="running",
        gates=[],
        protected_sha256=protected,
        links=a["links"],
        aot_caches=a["aot_caches"],
        boot_id=a["boot_id"],
        runtime_changed=False,
        host105_contacted=False,
    )
    cmd = [
        "podman",
        "run",
        "--rm",
        "--name",
        "bc250-ck-plugin-b071-build-" + args.attempt,
        "--network",
        "none",
        "--cpus",
        "4",
        "--memory",
        "10g",
        "--memory-swap",
        "12g",
    ]
    for src, dst, mode in (
        (WS / "source", "/workspace/source", "ro"),
        (WS / "helpers", "/workspace/helpers", "ro"),
        (WS / "build", "/workspace/build", "ro"),
        (WS / "cache", "/workspace/cache", "rw"),
        (output, "/workspace/output", "rw"),
    ):
        cmd += ["-v", f"{src}:{dst}:{mode}"]
    cmd += [
        "localhost/bc250-rocm-build:10.2-ccache",
        "bash",
        "/workspace/helpers/build_ck_factory_b071.sh",
    ]
    r["build_command"] = cmd
    atomic_json(logs / "progress.json", r)
    try:
        with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
            assert not verify_hashes(protected)
            with (logs / "build.log").open("w") as log:
                subprocess.run(
                    cmd, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=600
                )
            r["binary_sha256"] = {str(p): sha(p) for p in output.glob("factory-*")}
            assert len(r["binary_sha256"]) == 2
            for name, launcher, script, extra, count, profile in [
                (
                    "daily",
                    "bc250-torch",
                    "validate_ck_factory_b071.py",
                    ["--attempt", args.attempt],
                    3,
                    "daily",
                ),
                (
                    "rccl",
                    "bc250-torch",
                    "validate_ck_factory_b071.py",
                    ["--attempt", args.attempt],
                    3,
                    "rccl-single",
                ),
                ("core", "bc250-train", "validate_cleanup_runtime.py", [], 7, "daily"),
                (
                    "daily-consumers",
                    "bc250-train",
                    "validate_rccl_b038.py",
                    ["--daily", "--consumers"],
                    5,
                    "daily",
                ),
                (
                    "rccl-consumers",
                    "bc250-train",
                    "validate_rccl_b038.py",
                    ["--consumers"],
                    6,
                    "rccl-single",
                ),
            ]:
                label = "ck-plugin-b071-" + args.attempt + "-" + name
                scratch = base / label
                scratch.mkdir()
                os.chown(scratch, runtime["uid"], runtime["gid"])
                command = ["/usr/local/bin/" + launcher, "--runtime-profile", profile]
                for key, val in dict(
                    TMPDIR=str(scratch),
                    TMP=str(scratch),
                    TEMP=str(scratch),
                    MIOPEN_USER_DB_PATH=str(scratch / "miopen-db"),
                    MIOPEN_CUSTOM_CACHE_DIR=str(scratch / "miopen-cache"),
                ).items():
                    command += ["--env", key + "=" + val]
                command += [str(WS / "helpers" / script), *extra]
                r["gates"].append(gate(label, command, count, runtime))
                atomic_json(logs / "progress.json", r)
            r["status"] = "passed"
    except BaseException as error:
        r.update(status="failed", error=repr(error))
        raise
    finally:
        after = subprocess.check_output(["dmesg"])
        (logs / "kernel-after.log").write_bytes(after)
        r.update(
            changed_hashes=verify_hashes(protected),
            links_unchanged=all(os.readlink(p) == v for p, v in a["links"].items()),
            kernel_log_unchanged=before == after,
            running_containers=subprocess.check_output(
                ["podman", "ps", "-q"], text=True
            ).split(),
        )
        if (
            r["changed_hashes"]
            or not r["links_unchanged"]
            or before != after
            or r["running_containers"]
        ):
            r["status"] = "failed"
        atomic_json(logs / "final.json", r)
    assert r["status"] == "passed" and sum(x["checks"] for x in r["gates"]) == 24
    print("CK_PLUGIN_B071_ADMISSION_PASS", flush=True)


if __name__ == "__main__":
    main()
