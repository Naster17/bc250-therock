#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Qualify archive relocation with only extracted SDKs, runtime and owned fixtures mounted."""
import argparse
import re
import json
import os
from pathlib import Path
import pwd
import subprocess

from bc250_torch import LaunchOptions, build_command, runtime_lock
from build_migraphx_b067_on_host import verify_hashes
from deploy_torch_a04_on_host import atomic_json, sha
from package_migraphx_b067_on_host import STORAGE as DEFAULT_STORAGE
from qualify_rccl_b038_on_host import gate
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--consumer", type=Path, required=True)
    parser.add_argument("--storage", type=Path, default=DEFAULT_STORAGE)
    parser.add_argument("--external-receipt", type=Path)
    args = parser.parse_args()
    assert re.fullmatch(r"r[1-9][0-9]*", args.attempt)
    storage = args.storage
    preflight("migraphx-b067-qualification")
    assert args.package.parent == storage
    packaged = json.loads((args.package / "final.json").read_text())
    assert packaged["status"] == "packaged-not-qualified"
    assert sha(Path(packaged["archive"])) == packaged["archive_sha256"]
    assert args.consumer.is_file() and args.consumer.is_relative_to(storage)
    baseline = json.loads(
        (WS / "logs/migraphx-b066-qualification-r2/final.json").read_text()
    )
    external = WS / "tmp/migraphx-b064/20261025-storage/mount"
    protected = dict(baseline["protected_sha256"])
    external_receipt = None
    if args.external_receipt:
        external_expected = {
            p: d for p, d in protected.items() if Path(p).is_relative_to(external)
        }
        external_receipt = json.loads(args.external_receipt.read_text())
        assert external_receipt["status"] == "passed"
        assert external_receipt["verified_sha256"] == external_expected
        protected = {p: d for p, d in protected.items() if p not in external_expected}
        protected[str(args.external_receipt)] = sha(args.external_receipt)

    for path in (WS / "helpers").glob("*migraphx*b067*"):
        if path.is_file():
            protected[str(path)] = sha(path)
    protected[str(args.consumer)] = sha(args.consumer)
    protected[packaged["archive"]] = packaged["archive_sha256"]
    links = dict(baseline["links"])
    for name in packaged["roots"]:
        sdk = Path(name)
        index = json.loads((sdk / "share/bc250-sdk/sha256.json").read_text())
        for filename, digest in index["files"].items():
            protected[str(sdk / filename)] = digest
        for filename, target in index["symlinks"].items():
            assert os.readlink(sdk / filename) == target
            links[str(sdk / filename)] = target
    logs = WS / ("logs/migraphx-b067-qualification-" + args.attempt)
    logs.mkdir()
    manifest = json.loads(Path("/opt/bc250-stack/current/runtime.json").read_text())
    release = Path("/opt/bc250-stack/current").resolve(strict=True)
    groups = os.getgrouplist(pwd.getpwuid(manifest["uid"]).pw_name, manifest["gid"])
    groups += [Path(d).stat().st_gid for d in ("/dev/kfd", "/dev/dri/renderD128")]
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
        package=packaged,
        external_preservation=external_receipt,
        storage=str(storage),
    )
    try:
        with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
            assert not verify_hashes(protected)
            for number, name in enumerate(packaged["roots"]):
                sdk = Path(name)
                for kind, count in (("python", 50), ("consumer", 6)):
                    scratch = storage / (
                        "runtime-" + args.attempt + "-" + str(number) + "-" + kind
                    )
                    scratch.mkdir()
                    os.chown(scratch, manifest["uid"], manifest["gid"])
                    label = (
                        "migraphx-b067-" + args.attempt + "-" + str(number) + "-" + kind
                    )
                    target_logs = WS / "logs" / label
                    envfile = scratch / "environment"
                    env = {
                        "TMPDIR": str(scratch),
                        "TMP": str(scratch),
                        "TEMP": str(scratch),
                        "MIOPEN_USER_DB_PATH": str(scratch / "miopen-db"),
                        "MIOPEN_CUSTOM_CACHE_DIR": str(scratch / "miopen-cache"),
                    }
                    envfile.write_text(
                        "".join(k + "=" + v + "\n" for k, v in env.items())
                    )
                    script = (
                        "validate_migraphx_b067.py"
                        if kind == "python"
                        else "validate_migraphx_b067_consumer.py"
                    )
                    arguments = [
                        str(sdk / "bin/bc250-migraphx.py"),
                        str(WS / "helpers" / script),
                    ]
                    arguments += (
                        ["--order", "migraphx" if number == 0 else "ort"]
                        if kind == "python"
                        else ["--consumer", str(args.consumer)]
                    )
                    options = LaunchOptions(
                        arguments=arguments,
                        mounts=[
                            (str(sdk), "ro"),
                            (str(WS / "helpers"), "ro"),
                            (str(target_logs), "rw"),
                            (str(args.consumer), "ro"),
                        ],
                        environment=env,
                    )
                    # gate creates/chowns its output directory before executing.
                    # build_command validates mounts, so prepare that exact directory first.
                    target_logs.mkdir()
                    os.chown(target_logs, manifest["uid"], manifest["gid"])
                    command = build_command(
                        release, manifest, scratch, scratch, options, groups, envfile
                    )
                    target_logs.rmdir()
                    report["gates"].append(gate(label, command, count, manifest))
                    atomic_json(logs / "progress.json", report)
                    print("PASS", label, count, flush=True)
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
                        "migraphx-b067-" + args.attempt + "-" + name,
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
        report["links_unchanged"] = all(os.readlink(p) == v for p, v in links.items())
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
        and sum(g["checks"] for g in report["gates"]) == 134
    )


if __name__ == "__main__":
    main()
