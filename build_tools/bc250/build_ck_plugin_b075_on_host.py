#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build only the private per-ID probe against the qualified MIOpen cohort."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from bc250_torch import runtime_lock
from build_migraphx_b067_on_host import verify_hashes
from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--attempt",
        choices=("r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8"),
        required=True,
    )
    args = p.parse_args()
    preflight("ck-plugin-b075-build-" + args.attempt)
    assert os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize > 4 * 1024**3
    base = WS / "tmp/ck-plugin-b075"
    base.mkdir(exist_ok=True)
    out = base / ("build-" + args.attempt)
    out.mkdir()
    logs = WS / ("logs/ck-plugin-b075-build-" + args.attempt)
    logs.mkdir()
    a = json.loads((WS / "logs/ck-plugin-b074-qualification-r1/final.json").read_text())
    protected = dict(a["protected_sha256"])

    cfg = WS / "build/ml-libs/MIOpen/build/include/miopen/config.h"
    protected[str(cfg)] = sha(cfg)
    inputs = base / ("inputs-" + args.attempt)
    for path in inputs.iterdir():
        assert path.is_file()
        protected[str(path)] = sha(path)
    for n in ("build_ck_plugin_probe_b075.sh",):
        protected[str(WS / "helpers" / n)] = sha(WS / "helpers" / n)
    before = subprocess.check_output(["dmesg"])
    (logs / "kernel-before.log").write_bytes(before)
    cmd = [
        "podman",
        "run",
        "--rm",
        "--name",
        "bc250-ck-plugin-b075-" + args.attempt,
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
        (out, "/workspace/output", "rw"),
        (inputs, "/workspace/inputs", "ro"),
    ):
        cmd += ["-v", f"{src}:{dst}:{mode}"]
    cmd += [
        "localhost/bc250-rocm-build:10.2-ccache",
        "bash",
        "/workspace/helpers/build_ck_plugin_probe_b075.sh",
    ]
    r = dict(
        status="running",
        command=cmd,
        protected_sha256=protected,
        links=a["links"],
        aot_caches=a["aot_caches"],
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        runtime_changed=False,
        host105_contacted=False,
    )
    atomic_json(logs / "progress.json", r)
    try:
        with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
            assert not verify_hashes(protected)
            with (logs / "build.log").open("w") as f:
                subprocess.run(
                    cmd, check=True, stdout=f, stderr=subprocess.STDOUT, timeout=600
                )
            plugin = (
                WS / "tmp/ck-plugin-b073/build-r2/libMIOpenCKGroupedConv_gfx1013.so"
            )
            r.update(
                status="built-private-not-qualified",
                plugin=str(plugin),
                plugin_sha256=sha(plugin),
                binary_sha256={str(p): sha(p) for p in out.glob("probe-*")},
            )
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
    assert r["status"] == "built-private-not-qualified"
    print("CK_PLUGIN_B075_BUILD_PASS", flush=True)


if __name__ == "__main__":
    main()
