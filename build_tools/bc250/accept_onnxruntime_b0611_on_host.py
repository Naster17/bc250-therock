#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exercise public ORT opt-in, identity refusals/recovery and retained defaults."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import bc250_ort as activation
from bc250_torch import LaunchOptions, runtime_lock
from build_migraphx_b067_on_host import verify_hashes
from deploy_torch_a04_on_host import atomic_json, sha
from qualify_rccl_b038_on_host import gate
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight


def cpu_checks(logs):
    checks = []

    def run(name, command, code=0):
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=90, cwd=WS
        )
        (logs / (name + ".log")).write_text(result.stdout + result.stderr)
        assert result.returncode == code, (name, result.returncode, result.stderr)
        checks.append(name)
        return result

    public = "/usr/local/bin/bc250-ort"
    run("info", [public, "--ort-info"])
    for name, args in [
        ("shell", ["--shell"]),
        ("profile", ["--runtime-profile", "other", "-c", "pass"]),
        ("protected-env", ["--env", "LD_LIBRARY_PATH=/tmp", "-c", "pass"]),
        ("empty", []),
    ]:
        run("refuse-" + name, [public, *args], 2)
    deploy = [sys.executable, str(WS / "helpers/deploy_onnxruntime_b0611_on_host.py")]
    run(
        "unregister",
        [*deploy, "unregister", "--receipt", str(logs / "unregister.json")],
    )
    try:
        run("unregistered-refusal", [public, "--ort-info"], 2)
    finally:
        run("register", [*deploy, "register", "--receipt", str(logs / "register.json")])
    with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
        run(
            "updater-excluded",
            ["flock", "-n", "-x", "/opt/bc250-stack/current/.runtime.lock", "true"],
            1,
        )
        original = activation.REGISTRY
        temporary = activation.BASE / ".b0611-refusal-registration.json"
        record = json.loads(original.read_text())
        for name in ("index", "sdk"):
            changed = json.loads(json.dumps(record))
            if name == "index":
                changed["index_sha256"] = "0" * 64
            else:
                changed["sdk_registration"]["runtime_image"] = "0" * 64
            atomic_json(temporary, changed)
            activation.REGISTRY = temporary
            try:
                try:
                    activation.registered(LaunchOptions())
                except ValueError as error:
                    assert "identity mismatch" in str(error)
                    checks.append(name + "-identity-drift-refused")
                else:
                    raise AssertionError("Identity drift accepted")
            finally:
                activation.REGISTRY = original
        code = "import onnxruntime as o,os; assert o.__version__=='1.31.0'; assert 'MIGraphXExecutionProvider' not in o.get_available_providers(); assert 'BC250_ORT_PACKAGE' not in os.environ"
        for profile in ("daily", "rccl-single"):
            run(
                "default-cpu-preserved-" + profile,
                [
                    "/usr/local/bin/bc250-torch",
                    "--runtime-profile",
                    profile,
                    "-c",
                    code,
                ],
            )
        run("sdk-default-preserved", ["/usr/local/bin/bc250-migraphx", "-c", code])
    assert len(checks) == 14
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attempt", required=True)
    args = p.parse_args()
    assert args.attempt in ("r1", "r2", "r3")
    preflight("onnxruntime-b0611-acceptance-" + args.attempt)
    logs = WS / ("logs/onnxruntime-b0611-acceptance-" + args.attempt)
    logs.mkdir()
    baseline = json.loads(
        (WS / "logs/onnxruntime-b0611-relocation-r1/final.json").read_text()
    )
    protected = dict(baseline["protected_sha256"])
    links = dict(baseline["links"])
    for path in (WS / "helpers").glob("*b0611*"):
        if path.is_file():
            protected[str(path)] = sha(path)
    for path in [
        activation.REGISTRY,
        Path("/usr/local/bin/bc250-ort").resolve(),
        Path("/opt/bc250-stack/tools/ort-b0611/bc250_migraphx.py"),
        Path("/opt/bc250-stack/tools/ort-b0611/bc250_torch.py"),
    ]:
        protected[str(path)] = sha(path)
    record, root = activation.registered(LaunchOptions())
    index = json.loads((root / "share/bc250-ort/sha256.json").read_text())
    for n, d in index["files"].items():
        protected[str(root / n)] = d
    protected[str(root / "share/bc250-ort/sha256.json")] = record["index_sha256"]
    for n, v in index["symlinks"].items():
        links[str(root / n)] = v
    before = subprocess.check_output(["dmesg"])
    (logs / "kernel-before.log").write_bytes(before)
    manifest = json.loads(Path("/opt/bc250-stack/current/runtime.json").read_text())
    report = dict(
        status="running",
        gates=[],
        cpu_checks=[],
        protected_sha256=protected,
        links=links,
        aot_caches=baseline["aot_caches"],
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        runtime_changed=False,
        host105_contacted=False,
    )
    try:
        assert not verify_hashes(protected)
        report["cpu_checks"] = cpu_checks(logs)
        atomic_json(logs / "progress.json", report)
        with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
            for profile, order in [("daily", "ort"), ("rccl-single", "migraphx")]:
                label = "onnxruntime-b0611-" + args.attempt + "-installed-" + profile
                scratch = WS / "tmp/onnxruntime-b0611" / label
                scratch.mkdir()
                os.chown(scratch, manifest["uid"], manifest["gid"])
                command = ["/usr/local/bin/bc250-ort", "--runtime-profile", profile]
                for name, value in dict(
                    TMPDIR=str(scratch),
                    TMP=str(scratch),
                    TEMP=str(scratch),
                    MIOPEN_USER_DB_PATH=str(scratch / "miopen-db"),
                    MIOPEN_CUSTOM_CACHE_DIR=str(scratch / "miopen-cache"),
                ).items():
                    command += ["--env", name + "=" + value]
                command += [
                    str(WS / "helpers/validate_onnxruntime_b0611_installed.py"),
                    "--order",
                    order,
                ]
                report["gates"].append(gate(label, command, 34, manifest))
                atomic_json(logs / "progress.json", report)
                print("PASS", label, 34, flush=True)
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
                        "onnxruntime-b0611-" + args.attempt + "-" + name,
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
        report["status"] == "passed" and sum(g["checks"] for g in report["gates"]) == 90
    )
    print("ORT_B0611_INSTALLED_ACCEPTED", flush=True)


if __name__ == "__main__":
    main()
