#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify optional registration rollback, ordinary activation and untouched defaults."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import bc250_ck as activation
from bc250_torch import LaunchOptions, runtime_lock
from build_migraphx_b067_on_host import verify_hashes
from deploy_torch_a04_on_host import atomic_json, sha
from qualify_rccl_b038_on_host import gate
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight


def cpu_checks(logs):
    checks = []

    def run(label, command, code=0):
        p = subprocess.run(
            command,
            cwd=WS,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=90,
        )
        (logs / (label + ".log")).write_text(p.stdout + p.stderr)
        assert p.returncode == code, (label, p.stderr)
        checks.append(label)

    wrapper = "/usr/local/bin/bc250-ck"
    run("info", [wrapper, "--ck-info"])
    for label, args in [
        ("shell", ["--shell"]),
        ("empty", []),
        ("profile", ["--runtime-profile", "other", "missing.py"]),
        ("protected", ["--env", "LD_LIBRARY_PATH=/tmp", "missing.py"]),
        ("python-c", ["-c", "pass"]),
    ]:
        run("refuse-" + label, [wrapper, *args], 2)
    deploy = [sys.executable, str(WS / "helpers/deploy_ck_plugin_b078_on_host.py")]
    run(
        "unregister",
        [*deploy, "unregister", "--receipt", str(logs / "unregister.json")],
    )
    try:
        run("unregistered-refusal", [wrapper, "--ck-info"], 2)
    finally:
        run("register", [*deploy, "register", "--receipt", str(logs / "register.json")])
    with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
        run(
            "exclusive-updater-excluded",
            ["flock", "-n", "-x", "/opt/bc250-stack/current/.runtime.lock", "true"],
            1,
        )
        record = json.loads(activation.REGISTRY.read_text())
        path = activation.BASE / ".b078-bad-registry.json"
        record["runtime_sha256"]["daily"] = "0" * 64
        atomic_json(path, record)
        original = activation.REGISTRY
        activation.REGISTRY = path
        try:
            try:
                activation.registered(LaunchOptions())
            except ValueError as error:
                assert "runtime identity mismatch" in str(error)
                checks.append("runtime-identity-drift-refused")
            else:
                raise AssertionError("runtime drift accepted")
        finally:
            activation.REGISTRY = original
        scratch = activation.BASE / ".b078-bad-package"
        shutil.copytree(activation.BASE / activation.NAME, scratch)
        record = json.loads(activation.REGISTRY.read_text())
        activation.check_package(scratch, record)
        marker = scratch / "README.md"
        old = marker.read_bytes()
        marker.write_bytes(old + b"owned drift\n")
        try:
            activation.check_package(scratch, record)
        except ValueError as error:
            assert "payload identity mismatch" in str(error)
            checks.append("package-byte-drift-refused")
        else:
            raise AssertionError("payload drift accepted")
        marker.write_bytes(old)
        (scratch / "unexpected.py").write_text("# owned refusal\n")
        try:
            activation.check_package(scratch, record)
        except ValueError as error:
            assert "file set mismatch" in str(error)
            checks.append("package-extra-file-refused")
        else:
            raise AssertionError("extra file accepted")
    return checks


def main():
    preflight("ck-plugin-b078")
    logs = WS / "logs/ck-plugin-b078-qualification-r1"
    logs.mkdir()
    baseline = json.loads(
        (WS / "logs/ck-plugin-b077-qualification-r3/final.json").read_text()
    )
    protected = dict(baseline["protected_sha256"])
    record = json.loads(activation.REGISTRY.read_text())
    root = Path(record["package"])
    manifest = json.loads((root / "manifest.json").read_text())
    protected.update({str(root / n): h for n, h in manifest["files"].items()})
    protected[str(root / "manifest.json")] = sha(root / "manifest.json")
    protected[str(activation.REGISTRY)] = sha(activation.REGISTRY)
    for name, h in record["launcher_sha256"].items():
        protected[str(Path("/opt/bc250-stack/tools/ck-b078") / name)] = h
    for p in (WS / "helpers").glob("*b078*"):
        if p.is_file():
            protected[str(p)] = sha(p)
    protected[str(WS / "helpers/bc250_ck.py")] = sha(WS / "helpers/bc250_ck.py")
    links = dict(baseline["links"])
    links["/usr/local/bin/bc250-ck"] = os.readlink("/usr/local/bin/bc250-ck")
    before = subprocess.check_output(["dmesg"])
    (logs / "kernel-before.log").write_bytes(before)
    runtime = json.loads(Path("/opt/bc250-stack/current/runtime.json").read_text())
    r = dict(
        status="running",
        gates=[],
        cpu_checks=[],
        protected_sha256=protected,
        links=links,
        aot_caches=baseline["aot_caches"],
        boot_id=baseline["boot_id"],
        native_runtime_changed=False,
        host105_contacted=False,
    )
    try:
        assert not verify_hashes(protected)
        # CPU-only registration operations acquire their own exclusive locks.
        r["cpu_checks"] = cpu_checks(logs)
        atomic_json(logs / "progress.json", r)
        with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
            for label, launcher, script, extra, count, profile in [
                (
                    "host-cli",
                    None,
                    "validate_ck_plugin_b078_cli_on_host.py",
                    [],
                    6,
                    "daily",
                ),
                (
                    "installed-daily",
                    "bc250-ck",
                    "validate_ck_plugin_b078.py",
                    [],
                    10,
                    "daily",
                ),
                (
                    "installed-rccl",
                    "bc250-ck",
                    "validate_ck_plugin_b078.py",
                    [],
                    10,
                    "rccl-single",
                ),
                (
                    "plain-daily",
                    "bc250-torch",
                    "validate_ck_plugin_b078.py",
                    ["--plain"],
                    6,
                    "daily",
                ),
                (
                    "plain-rccl",
                    "bc250-torch",
                    "validate_ck_plugin_b078.py",
                    ["--plain"],
                    6,
                    "rccl-single",
                ),
                ("core", "bc250-train", "validate_cleanup_runtime.py", [], 7, "daily"),
                (
                    "onnx",
                    "bc250-torch",
                    "validate_onnx_b061.py",
                    ["--group", "audit"],
                    4,
                    "daily",
                ),
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
                name = "ck-plugin-b078-r1-" + label
                scratch = WS / "tmp/ck-plugin-b078" / name
                scratch.mkdir(parents=True)
                os.chown(scratch, runtime["uid"], runtime["gid"])
                if launcher is None:
                    cmd = [sys.executable, str(WS / "helpers" / script), *extra]
                else:
                    cmd = ["/usr/local/bin/" + launcher, "--runtime-profile", profile]
                    env = dict(
                        TMPDIR=str(scratch),
                        TMP=str(scratch),
                        TEMP=str(scratch),
                        MIOPEN_USER_DB_PATH=str(scratch / "db"),
                        MIOPEN_CUSTOM_CACHE_DIR=str(scratch / "cache"),
                    )
                    if launcher == "bc250-ck":
                        env.update(BC250_CK_MARKER="installed", MIOPEN_LOG_LEVEL="6")
                    for k, v in env.items():
                        cmd += ["--env", k + "=" + v]
                    cmd += [str(WS / "helpers" / script), *extra]
                r["gates"].append(gate(name, cmd, count, runtime))
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
            links_unchanged=all(os.readlink(p) == v for p, v in links.items()),
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
    assert r["status"] == "passed" and sum(g["checks"] for g in r["gates"]) == 60
    print("CK_PLUGIN_B078_QUALIFICATION_PASS", flush=True)


if __name__ == "__main__":
    main()
