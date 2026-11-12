#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify B07.13 activation, identity/refusals, live rollback and untouched defaults.

The B07.12 qualification final.json is used only as artifact-qualification
proof: its protected set/links/boot/aot_caches predate the D61 release, so a
fresh current-state baseline is captured here instead of inherited. 18 CPU
registration/identity/CLI/lock/rollback checks plus nine guarded gates (72
checks); the separate ordinary-user guard adds 2 checks (92 total, 10
acknowledged guards). The retained B07.7 package/tools/record are never
removed; rollback restores the captured pre-activation state exactly.
"""
import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import bc250_ck_b0713 as activation
from bc250_torch import LaunchOptions, runtime_lock, selected_release
from build_migraphx_b067_on_host import verify_hashes
from deploy_torch_a04_on_host import atomic_json, sha
from qualify_rccl_b038_on_host import gate
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight

PREVIOUS_NAME = "bc250-miopen-dl-gfx1013-b077"
ROLLBACK = activation.BASE / "miopen-dl-b0713-rollback.json"
GATE_CHECKS = 72


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
        return p

    wrapper = "/usr/local/bin/bc250-ck"
    proc = run("info", [wrapper, "--ck-info"])
    record = json.loads(proc.stdout)
    assert record["package"] == str(activation.BASE / activation.NAME)
    for label, args in [
        ("shell", ["--shell"]),
        ("empty", []),
        ("profile", ["--runtime-profile", "other", "missing.py"]),
        ("protected", ["--env", "LD_LIBRARY_PATH=/tmp", "missing.py"]),
        ("python-c", ["-c", "pass"]),
    ]:
        run("refuse-" + label, [wrapper, *args], 2)
    deploy = [sys.executable, str(WS / "helpers/deploy_ck_plugin_b0713_on_host.py")]
    run(
        "unregister",
        [*deploy, "unregister", "--receipt", str(logs / "unregister.json")],
    )
    try:
        run("unregistered-refusal", [wrapper, "--ck-info"], 2)
    finally:
        run("activate", [*deploy, "activate", "--receipt", str(logs / "activate.json")])
    with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
        run(
            "exclusive-updater-excluded",
            ["flock", "-n", "-x", "/opt/bc250-stack/current/.runtime.lock", "true"],
            1,
        )
        record = json.loads(activation.REGISTRY.read_text())
        path = activation.BASE / ".b0713-bad-registry.json"
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
        scratch = activation.BASE / ".b0713-bad-package"
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
        # Retire the owned drift fixtures; they are not part of the protected set.
        shutil.rmtree(scratch)
        path.unlink()
    # Live rollback: restore the captured pre-activation registry bytes and
    # wrapper symlink, prove the old activation, then re-activate b0712. The
    # retained old wrapper may read its own cohort registry, so the b077
    # identity is asserted on the restored live registry bytes themselves.
    captured = json.loads(ROLLBACK.read_text())
    run("rollback", [*deploy, "rollback", "--receipt", str(logs / "rollback.json")])
    assert activation.REGISTRY.read_bytes() == base64.b64decode(
        captured["registry_bytes_b64"]
    )
    restored = json.loads(activation.REGISTRY.read_text())
    assert restored["package"] == str(activation.BASE / PREVIOUS_NAME)
    assert os.readlink(wrapper) == captured["symlink_target"]
    checks.append("rollback-b077-registry-and-wrapper-restored")
    proc = run("rollback-ck-info", [wrapper, "--ck-info"])
    old_record = json.loads(proc.stdout)
    assert old_record["package"] != str(activation.BASE / activation.NAME)
    run(
        "reactivate",
        [*deploy, "activate", "--receipt", str(logs / "reactivate.json")],
    )
    proc = run("reactivated-ck-info", [wrapper, "--ck-info"])
    record = json.loads(proc.stdout)
    assert record["package"] == str(activation.BASE / activation.NAME)
    assert record["archive_sha256"] == activation.ARCHIVE_SHA
    return checks


def main():
    preflight("ck-plugin-b0713")
    logs = WS / "logs/ck-plugin-b0713-qualification-r1"
    logs.mkdir()
    artifact = json.loads(
        (WS / "logs/ck-plugin-b0712-qualification-r2/final.json").read_text()
    )
    assert artifact["status"] == "passed"
    record = json.loads(activation.REGISTRY.read_text())
    root = Path(record["package"])
    assert root == activation.BASE / activation.NAME
    manifest = json.loads((root / "manifest.json").read_text())
    protected = {str(root / n): h for n, h in manifest["files"].items()}
    protected[str(root / "manifest.json")] = sha(root / "manifest.json")
    # Retained B07.7 rollback target: package payload, manifest, version record.
    previous = activation.BASE / PREVIOUS_NAME
    previous_manifest = json.loads((previous / "manifest.json").read_text())
    protected.update(
        {str(previous / n): h for n, h in previous_manifest["files"].items()}
    )
    protected[str(previous / "manifest.json")] = sha(previous / "manifest.json")
    for name in (PREVIOUS_NAME, activation.NAME):
        path = activation.BASE / (name + ".json")
        protected[str(path)] = sha(path)
    protected[str(activation.REGISTRY)] = sha(activation.REGISTRY)
    for tools in (
        Path("/opt/bc250-stack/tools/ck-b078"),
        Path("/opt/bc250-stack/tools/ck-b0713"),
    ):
        for p in sorted(tools.iterdir()):
            if p.is_file():
                protected[str(p)] = sha(p)
    # Fresh native cohort: hash the selected-release files the artifact maps
    # (container /workspace/build/dist/rocm -> release/rocm). Image-local
    # system libraries stay pinned by the recorded runtime images/manifests.
    container_rocm = Path("/workspace/build/dist/rocm")
    for profile in ("daily", "rccl-single"):
        release = selected_release(LaunchOptions(runtime_profile=profile))
        protected[str(release / "runtime.json")] = sha(release / "runtime.json")
        for entry in manifest["runtime_libraries"].values():
            path = Path(entry["path"])
            if path.is_relative_to(container_rocm):
                host = release / "rocm" / path.relative_to(container_rocm)
                assert sha(host) == entry["sha256"], host
                protected[str(host)] = entry["sha256"]
    for p in (WS / "helpers").glob("*b0713*"):
        if p.is_file():
            protected[str(p)] = sha(p)
    for name in (
        "bc250_ck_b0713.py",
        "bc250_ck.py",
        "bc250_torch.py",
        "bc250_migraphx.py",
    ):
        path = WS / "helpers" / name
        protected[str(path)] = sha(path)
    for path in (
        Path("/usr/local/bin/bc250-torch"),
        Path("/usr/local/bin/bc250-train"),
        Path("/usr/local/bin/bc250-ck").resolve(strict=True),
    ):
        protected[str(path)] = sha(path)
    # In the D61 release current/torch is a real directory (the patched Torch
    # source tree), not a symlink, so its payload identity is protected by hash
    # rather than readlink; the runtime-selection symlinks are checked below.
    torch_init = Path("/opt/bc250-stack/current/torch/torch/__init__.py")
    protected[str(torch_init)] = sha(torch_init)
    aot = (
        sorted((WS / "build/math-libs/rocFFT").rglob("*.db"))
        if (WS / "build/math-libs/rocFFT").is_dir()
        else []
    )
    for p in aot:
        protected[str(p)] = sha(p)
    links = {
        str(p): os.readlink(p)
        for p in (
            Path("/opt/bc250-stack/current"),
            Path("/opt/bc250-stack/previous"),
            Path("/usr/local/bin/bc250-ck"),
            Path("/opt/bc250-stack/profiles/rccl-single"),
            Path("/opt/bc250-stack/current/torch"),
            Path("/opt/bc250-stack/current/rocm"),
        )
        if os.path.islink(p)
    }
    before = subprocess.check_output(["dmesg"])
    (logs / "kernel-before.log").write_bytes(before)
    runtime = json.loads(Path("/opt/bc250-stack/current/runtime.json").read_text())
    r = dict(
        status="running",
        gates=[],
        cpu_checks=[],
        protected_sha256=protected,
        links=links,
        aot_caches=[str(p) for p in aot],
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
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
                    "validate_ck_plugin_b0713_cli_on_host.py",
                    [],
                    8,
                    "daily",
                ),
                (
                    "installed-daily",
                    "bc250-ck",
                    "validate_ck_plugin_b0713.py",
                    [],
                    15,
                    "daily",
                ),
                (
                    "installed-rccl",
                    "bc250-ck",
                    "validate_ck_plugin_b0713.py",
                    [],
                    15,
                    "rccl-single",
                ),
                (
                    "plain-daily",
                    "bc250-torch",
                    "validate_ck_plugin_b0713.py",
                    ["--plain"],
                    6,
                    "daily",
                ),
                (
                    "plain-rccl",
                    "bc250-torch",
                    "validate_ck_plugin_b0713.py",
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
                name = "ck-plugin-b0713-r1-" + label
                scratch = WS / "tmp/ck-plugin-b0713" / name
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
    assert (
        r["status"] == "passed" and sum(g["checks"] for g in r["gates"]) == GATE_CHECKS
    )
    print("CK_PLUGIN_B0713_QUALIFICATION_PASS", flush=True)


if __name__ == "__main__":
    main()
