#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Qualify two ORT extraction paths without original build/home trees mounted."""
import argparse
import json
import os
from pathlib import Path
import pwd
import subprocess

from bc250_migraphx import registered
from bc250_torch import LaunchOptions, build_command, runtime_lock
from build_migraphx_b067_on_host import verify_hashes
from deploy_torch_a04_on_host import atomic_json, sha
from package_onnxruntime_b0611_on_host import BASE
from qualify_rccl_b038_on_host import gate
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attempt", required=True)
    args = p.parse_args()
    assert args.attempt in ("r1", "r2", "r3")
    preflight("onnxruntime-b0611-relocation-" + args.attempt)
    packaged = json.loads((BASE / "package-r1/final.json").read_text())
    assert sha(Path(packaged["archive"])) == packaged["archive_sha256"]
    baseline = json.loads(
        (WS / "logs/onnxruntime-b0610-qualification-r1/final.json").read_text()
    )
    protected = dict(baseline["protected_sha256"])
    links = dict(baseline["links"])
    for path in (WS / "helpers").glob("*b0611*"):
        if path.is_file():
            protected[str(path)] = sha(path)
    protected[packaged["archive"]] = packaged["archive_sha256"]
    for name in packaged["roots"]:
        root = Path(name)
        index = json.loads((root / "share/bc250-ort/sha256.json").read_text())
        for n, d in index["files"].items():
            protected[str(root / n)] = d
        protected[str(root / "share/bc250-ort/sha256.json")] = packaged["index_sha256"]
        for n, v in index["symlinks"].items():
            links[str(root / n)] = v
    logs = WS / ("logs/onnxruntime-b0611-relocation-" + args.attempt)
    logs.mkdir()
    release = Path("/opt/bc250-stack/current").resolve(strict=True)
    manifest = json.loads((release / "runtime.json").read_text())
    groups = os.getgrouplist(pwd.getpwuid(manifest["uid"]).pw_name, manifest["gid"])
    groups += [Path(p).stat().st_gid for p in ("/dev/kfd", "/dev/dri/renderD128")]
    before = subprocess.check_output(["dmesg"])
    (logs / "kernel-before.log").write_bytes(before)
    report = dict(
        status="running",
        gates=[],
        protected_sha256=protected,
        links=links,
        aot_caches=baseline["aot_caches"],
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        runtime_changed=False,
        host105_contacted=False,
    )
    try:
        with runtime_lock(release / ".runtime.lock"):
            assert not verify_hashes(protected)
            _, sdk = registered(LaunchOptions())
            for i, name in enumerate(packaged["roots"]):
                root = Path(name)
                label = "onnxruntime-b0611-" + args.attempt + "-relocated-" + str(i)
                scratch = BASE / label
                scratch.mkdir()
                os.chown(scratch, manifest["uid"], manifest["gid"])
                output = WS / "logs" / label
                output.mkdir()
                os.chown(output, manifest["uid"], manifest["gid"])
                environment = dict(
                    TMPDIR=str(scratch),
                    TMP=str(scratch),
                    TEMP=str(scratch),
                    MIOPEN_USER_DB_PATH=str(scratch / "miopen-db"),
                    MIOPEN_CUSTOM_CACHE_DIR=str(scratch / "miopen-cache"),
                )
                envfile = scratch / "environment"
                envfile.write_text(
                    "".join(k + "=" + v + "\n" for k, v in environment.items())
                )
                options = LaunchOptions(
                    arguments=[
                        str(sdk / "bin/bc250-migraphx.py"),
                        str(root / "bin/bc250-ort.py"),
                        str(WS / "helpers/validate_onnxruntime_b0611.py"),
                        "--order",
                        "ort" if i == 0 else "torch",
                    ],
                    mounts=[
                        (str(sdk), "ro"),
                        (str(root), "ro"),
                        (str(WS / "helpers"), "ro"),
                        (str(output), "rw"),
                    ],
                    environment=environment,
                )
                command = build_command(
                    release, manifest, scratch, scratch, options, groups, envfile
                )
                output.rmdir()
                report["gates"].append(gate(label, command, 34, manifest))
                atomic_json(logs / "progress.json", report)
                print("PASS", label, 34, flush=True)
            report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        after = subprocess.check_output(["dmesg"])
        (logs / "kernel-after.log").write_bytes(after)
        report.update(
            changed_hashes=verify_hashes(protected),
            links_unchanged=all(os.readlink(p) == v for p, v in links.items()),
            kernel_log_unchanged=before == after,
            running_containers=subprocess.check_output(
                ["podman", "ps", "-q"], text=True
            ).split(),
        )
        if (
            report["changed_hashes"]
            or not report["links_unchanged"]
            or before != after
            or report["running_containers"]
        ):
            report["status"] = "failed"
        atomic_json(logs / "final.json", report)
    assert (
        report["status"] == "passed" and sum(g["checks"] for g in report["gates"]) == 68
    )
    print("ORT_B0611_RELOCATION_PASS", flush=True)


if __name__ == "__main__":
    main()
