#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Guard bounded forced-solver paired timing without changing the active prefix."""
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
from board_preflight import preflight


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attempt", choices=("r1", "r2", "r3", "r4", "r5"), required=True)
    p.add_argument(
        "--build-attempt", choices=("r1", "r2", "r3", "r4", "r5"), required=True
    )
    args = p.parse_args()
    preflight("ck-plugin-b0711-" + args.attempt)
    logs = WS / ("logs/ck-plugin-b0711-qualification-" + args.attempt)
    logs.mkdir()
    a = json.loads(
        (
            WS / ("logs/ck-plugin-b0711-build-" + args.build_attempt + "/final.json")
        ).read_text()
    )
    assert a["status"] == "built-private-not-qualified"
    protected = dict(a["protected_sha256"])
    protected[a["plugin"]] = a["plugin_sha256"]
    root = WS / "tmp/ck-plugin-b0710/build-r2"
    protected.update(a["binary_sha256"])
    protected.update(a.get("object_sha256", {}))
    for path in (WS / "helpers").glob("*b0711*"):
        if path.is_file():
            protected[str(path)] = sha(path)
    native = json.loads(
        (WS / "tmp/ck-plugin-b0711/native-fix-r2/final.json").read_text()
    )
    assert native["status"] == "built-isolated-native-fix-not-qualified"
    protected.update(native["protected_sha256"])
    protected[native["library"]] = native["library_sha256"]
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
    try:
        with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
            assert not verify_hashes(protected)
            for name, script, extra, count, profile in [
                (
                    "private-daily",
                    "validate_ck_plugin_b0711.py",
                    ["--build-attempt", args.build_attempt],
                    7,
                    "daily",
                ),
                (
                    "private-rccl",
                    "validate_ck_plugin_b0711.py",
                    ["--build-attempt", args.build_attempt],
                    7,
                    "rccl-single",
                ),
                (
                    "private-contracts-daily",
                    "validate_ck_plugin_b0710.py",
                    ["--build-attempt", "r2"],
                    12,
                    "daily",
                ),
                (
                    "private-contracts-rccl",
                    "validate_ck_plugin_b0710.py",
                    ["--build-attempt", "r2"],
                    12,
                    "rccl-single",
                ),
                ("installed-daily", "validate_ck_plugin_b078.py", [], 10, "daily"),
                ("installed-rccl", "validate_ck_plugin_b078.py", [], 10, "rccl-single"),
                ("core", "validate_cleanup_runtime.py", [], 7, "daily"),
                ("onnx", "validate_onnx_b061.py", ["--group", "audit"], 4, "daily"),
                (
                    "daily-consumers",
                    "validate_rccl_b038.py",
                    ["--daily", "--consumers"],
                    5,
                    "daily",
                ),
                (
                    "rccl-consumers",
                    "validate_rccl_b038.py",
                    ["--consumers"],
                    6,
                    "rccl-single",
                ),
            ]:
                label = "ck-plugin-b0711-" + args.attempt + "-" + name
                scratch = WS / "tmp/ck-plugin-b0711" / label
                scratch.mkdir()
                os.chown(scratch, runtime["uid"], runtime["gid"])
                launcher = (
                    "bc250-train"
                    if name in ("core", "daily-consumers", "rccl-consumers")
                    else "bc250-torch"
                )
                if name.startswith("installed-"):
                    launcher = "bc250-ck"
                cmd = ["/usr/local/bin/" + launcher, "--runtime-profile", profile]
                env = dict(
                    TMPDIR=str(scratch),
                    TMP=str(scratch),
                    TEMP=str(scratch),
                    MIOPEN_USER_DB_PATH=str(scratch / "miopen-db"),
                    MIOPEN_CUSTOM_CACHE_DIR=str(scratch / "miopen-cache"),
                )
                if name.startswith("installed-"):
                    env.update(BC250_CK_MARKER="installed", MIOPEN_LOG_LEVEL="6")
                if name.startswith("private-"):
                    env.update(
                        MIOPEN_CK_LIB_PATH=str(root),
                        MIOPEN_LOG_LEVEL="6",
                    )
                for k, v in env.items():
                    cmd += ["--env", k + "=" + v]
                cmd += [str(WS / "helpers" / script), *extra]
                r["gates"].append(gate(label, cmd, count, runtime))
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
    assert r["status"] == "passed" and sum(g["checks"] for g in r["gates"]) == 80
    print("CK_PLUGIN_B0711_QUALIFICATION_PASS", flush=True)


if __name__ == "__main__":
    main()
