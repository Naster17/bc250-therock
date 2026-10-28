#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Qualify versioned SDK registration/recovery and both ordinary runtime cohorts."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import bc250_migraphx as activation
from bc250_torch import LaunchOptions, runtime_lock
from build_migraphx_b067_on_host import verify_hashes
from deploy_torch_a04_on_host import atomic_json, sha
from qualify_rccl_b038_on_host import gate
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight


def cpu_checks(logs):
    checks = []

    def run(name, command, code=0, contains=None):
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=60, cwd=WS
        )
        (logs / (name + ".log")).write_text(result.stdout + result.stderr)
        assert result.returncode == code, (name, result.returncode, result.stderr)
        if contains:
            assert contains in result.stdout + result.stderr
        checks.append(name)
        return result

    deploy = [sys.executable, str(WS / "helpers/deploy_migraphx_b068_on_host.py")]
    run(
        "info",
        ["/usr/local/bin/bc250-migraphx", "--sdk-info"],
        contains=activation.NAME,
    )
    for name, args in [
        ("shell", ["--shell"]),
        ("profile", ["--runtime-profile", "other", "-c", "pass"]),
        ("protected-env", ["--env", "LD_LIBRARY_PATH=/tmp", "-c", "pass"]),
        ("empty", []),
    ]:
        run("refuse-" + name, ["/usr/local/bin/bc250-migraphx", *args], code=2)
    run(
        "unregister",
        [*deploy, "unregister", "--receipt", str(logs / "unregister.json")],
    )
    try:
        run(
            "unregistered-refusal",
            ["/usr/local/bin/bc250-migraphx", "--sdk-info"],
            code=2,
            contains="migraphx.json",
        )
    finally:
        run("register", [*deploy, "register", "--receipt", str(logs / "register.json")])
    with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
        run(
            "exclusive-updater-excluded",
            ["flock", "-n", "-x", "/opt/bc250-stack/current/.runtime.lock", "true"],
            code=1,
        )
        # Root-owned scratch registration exercises fail-closed runtime identity
        # without modifying the live registration or a qualified runtime manifest.
        temporary = Path("/opt/bc250-stack/sdks/.b068-refusal-registration.json")
        record = json.loads(activation.REGISTRY.read_text())
        record["runtime_sha256"]["daily"] = "0" * 64
        atomic_json(temporary, record)
        original = activation.REGISTRY
        activation.REGISTRY = temporary
        try:
            try:
                activation.registered(LaunchOptions())
            except ValueError as error:
                assert "runtime identity mismatch" in str(error)
                checks.append("runtime-identity-drift-refused")
            else:
                raise AssertionError("Runtime drift accepted")
        finally:
            activation.REGISTRY = original
        scratch = Path("/opt/bc250-stack/sdks/.b068-refusal-sdk")
        shutil.copytree(activation.BASE / activation.NAME, scratch, symlinks=True)
        activation.check_sdk(scratch)
        marker = scratch / "README.md"
        original_bytes = marker.read_bytes()
        marker.write_bytes(original_bytes + b"identity drift\n")
        try:
            activation.check_sdk(scratch)
        except ValueError as error:
            assert "SDK file identity mismatch" in str(error)
            checks.append("sdk-byte-drift-refused")
        else:
            raise AssertionError("SDK drift accepted")
        marker.write_bytes(original_bytes)
        extra = scratch / "unexpected-module.py"
        extra.write_text("# owned refusal fixture\n")
        try:
            activation.check_sdk(scratch)
        except ValueError as error:
            assert "file set mismatch" in str(error)
            checks.append("sdk-extra-file-refused")
        else:
            raise AssertionError("Unexpected SDK file accepted")
        # Retain this deliberately invalid scratch fixture; no SDK/source deletion.
        for profile in ("daily", "rccl-single"):
            run(
                "default-sdk-absent-" + profile,
                [
                    "/usr/local/bin/bc250-torch",
                    "--runtime-profile",
                    profile,
                    "-c",
                    "import importlib.util,os; assert importlib.util.find_spec('migraphx') is None; assert 'BC250_MIGRAPHX_SDK' not in os.environ",
                ],
            )
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True)
    args = parser.parse_args()
    assert args.attempt in ("r1", "r2", "r3")
    preflight("migraphx-b068-qualification-" + args.attempt)
    logs = WS / ("logs/migraphx-b068-qualification-" + args.attempt)
    logs.mkdir()
    baseline = json.loads(
        (WS / "logs/migraphx-b067-qualification-r5/final.json").read_text()
    )
    protected = dict(baseline["protected_sha256"])
    for path in (WS / "helpers").glob("*migraphx*b068*"):
        if path.is_file():
            protected[str(path)] = sha(path)
    for path in (
        activation.REGISTRY,
        Path("/usr/local/bin/bc250-migraphx").resolve(),
        Path("/opt/bc250-stack/tools/migraphx-b068/bc250_torch.py"),
    ):
        protected[str(path)] = sha(path)
    before = subprocess.check_output(["dmesg"])
    (logs / "kernel-before.log").write_bytes(before)
    report = dict(
        status="running",
        gates=[],
        cpu_checks=[],
        protected_sha256=protected,
        links=baseline["links"],
        aot_caches=baseline["aot_caches"],
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        host105_contacted=False,
        active_runtime_changed=False,
    )
    manifest = json.loads(Path("/opt/bc250-stack/current/runtime.json").read_text())
    fixture = Path("/opt/bc250-stack/sdk-fixtures/b068/application with spaces")
    fixture.mkdir(parents=True)
    shutil.copyfile(WS / "helpers/validate_migraphx_b068.py", fixture / "activation.py")
    (fixture / "sidecar.json").write_text('{"owned_fixture": true}\n')
    try:
        assert not verify_hashes(protected)
        report["cpu_checks"] = cpu_checks(logs)
        atomic_json(logs / "progress.json", report)
        with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
            for profile in ("daily", "rccl-single"):
                for kind, script, extra, count in [
                    (
                        "activation",
                        str(fixture / "activation.py"),
                        ["--profile", profile],
                        8,
                    ),
                    (
                        "python",
                        str(WS / "helpers/validate_migraphx_b067.py"),
                        ["--order", "migraphx" if profile == "daily" else "torch"],
                        50,
                    ),
                    (
                        "consumer",
                        str(WS / "helpers/validate_migraphx_b067_consumer.py"),
                        [
                            "--consumer",
                            str(
                                WS
                                / "tmp/migraphx-b067-native/consumer-r2/build/consumer"
                            ),
                        ],
                        6,
                    ),
                ]:
                    label = "migraphx-b068-" + args.attempt + "-" + profile + "-" + kind
                    scratch = WS / "tmp/migraphx-b068" / label
                    scratch.mkdir(parents=True)
                    os.chown(scratch, manifest["uid"], manifest["gid"])
                    command = [
                        "/usr/local/bin/bc250-migraphx",
                        "--runtime-profile",
                        profile,
                        "--env",
                        "BC250_B068_MARKER=ordinary activation",
                    ]
                    for name, value in dict(
                        TMPDIR=str(scratch),
                        TMP=str(scratch),
                        TEMP=str(scratch),
                        MIOPEN_USER_DB_PATH=str(scratch / "miopen-db"),
                        MIOPEN_CUSTOM_CACHE_DIR=str(scratch / "miopen-cache"),
                    ).items():
                        command += ["--env", name + "=" + value]
                    command += [script, *extra]
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
                        "migraphx-b068-" + args.attempt + "-" + name,
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
        and sum(g["checks"] for g in report["gates"]) == 150
    )
    print("MIGRAPHX_B068_QUALIFIED", flush=True)


if __name__ == "__main__":
    main()
