#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Copy the existing MIOpen build and rebuild only the changed embedded kernel batch."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from bc250_torch import runtime_lock
from build_migraphx_b067_on_host import verify_hashes
from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS
from board_preflight import preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", choices=("r1", "r2", "r3"), required=True)
    args = parser.parse_args()
    preflight("ck-plugin-b0711-native-fix-" + args.attempt)
    base = WS / "tmp/ck-plugin-b0711"
    out = base / ("native-fix-" + args.attempt)
    out.mkdir()
    a = json.loads((WS / "logs/ck-plugin-b0711-build-r3/final.json").read_text())
    assert a["status"] == "built-private-not-qualified"
    protected = dict(a["protected_sha256"])
    patch = base / "native-inputs-r1/MIOpenConvDirUniHip.cpp"
    protected[str(patch)] = sha(patch)
    protected[str(WS / "helpers/build_miopen_b0711_fix.sh")] = sha(
        WS / "helpers/build_miopen_b0711_fix.sh"
    )
    original = WS / "build/ml-libs/MIOpen/build"
    # No hardlinks: rebuilding must never modify the preserved original inode.
    assert os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize > 12 * 1024**3
    before = subprocess.check_output(["dmesg"])
    (out / "kernel-before.log").write_bytes(before)
    r = dict(
        status="running",
        protected_sha256=protected,
        boot_id=a["boot_id"],
        active_runtime_changed=False,
        host105_contacted=False,
    )
    try:
        with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
            assert not verify_hashes(protected)
            target = out / "miopen-build"
            subprocess.run(
                ["cp", "-a", "--reflink=auto", str(original), str(target)],
                check=True,
                timeout=180,
            )
            assert (target / "lib/libMIOpen.so.1.0").stat().st_ino != (
                original / "lib/libMIOpen.so.1.0"
            ).stat().st_ino
            cmd = [
                "podman",
                "run",
                "--rm",
                "--name",
                "bc250-miopen-b0711-fix-" + args.attempt,
                "--network",
                "none",
                "--cpus",
                "4",
                "--memory",
                "10g",
                "--memory-swap",
                "12g",
            ]
            for src, dst, mode in [
                (WS / "source", "/workspace/source", "ro"),
                (WS / "helpers", "/workspace/helpers", "ro"),
                (WS / "build", "/workspace/build", "ro"),
                (WS / "cache", "/workspace/cache", "rw"),
                (out, "/workspace/output", "rw"),
                (target, "/workspace/build/ml-libs/MIOpen/build", "rw"),
                (
                    patch,
                    "/workspace/source/rocm-libraries/projects/miopen/src/kernels/MIOpenConvDirUniHip.cpp",
                    "ro",
                ),
            ]:
                cmd += ["-v", f"{src}:{dst}:{mode}"]
            cmd += [
                "localhost/bc250-rocm-build:10.2-ccache",
                "bash",
                "/workspace/helpers/build_miopen_b0711_fix.sh",
            ]
            r["command"] = cmd
            atomic_json(out / "progress.json", r)
            with (out / "controller-build.log").open("w") as log:
                subprocess.run(
                    cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=600,
                )
            library = target / "lib/libMIOpen.so.1.0"
            r.update(
                status="built-isolated-native-fix-not-qualified",
                library=str(library),
                library_sha256=sha(library),
            )
    except BaseException as error:
        r.update(status="failed", error=repr(error))
        raise
    finally:
        after = subprocess.check_output(["dmesg"])
        (out / "kernel-after.log").write_bytes(after)
        r.update(
            changed_hashes=verify_hashes(protected),
            kernel_log_unchanged=before == after,
            running_containers=subprocess.check_output(
                ["podman", "ps", "-q"], text=True
            ).split(),
        )
        if r["changed_hashes"] or before != after or r["running_containers"]:
            r["status"] = "failed"
        atomic_json(out / "final.json", r)
    assert r["status"] == "built-isolated-native-fix-not-qualified"
    print("MIOPEN_B0711_NATIVE_FIX_BUILD_PASS", flush=True)


if __name__ == "__main__":
    main()
