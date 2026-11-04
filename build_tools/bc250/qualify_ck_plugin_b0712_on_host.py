#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Qualify two extracted plugin artifacts with original candidate/build paths unmounted."""
import argparse
import json
import os
from pathlib import Path
import pwd
import shutil
import subprocess
from bc250_torch import LaunchOptions, build_command, runtime_lock, selected_release
from build_migraphx_b067_on_host import verify_hashes
from deploy_torch_a04_on_host import atomic_json, sha
from qualify_rccl_b038_on_host import gate
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", choices=("r1", "r2", "r3"), required=True)
    parser.add_argument("--package-attempt", choices=("r1", "r2", "r3"), required=True)
    args = parser.parse_args()
    preflight("ck-plugin-b0712-" + args.attempt)
    base = WS / "tmp/ck-plugin-b0712"
    a = json.loads(
        (base / ("package-" + args.package_attempt + "/final.json")).read_text()
    )
    assert a["status"] == "packaged-not-qualified"
    protected = dict(a["protected_sha256"])
    protected[a["archive"]] = a["archive_sha256"]
    for folder in a["roots"]:
        root = Path(folder)
        data = json.loads((root / "manifest.json").read_text())
        protected[str(root / "manifest.json")] = sha(root / "manifest.json")
        protected.update({str(root / k): v for k, v in data["files"].items()})
    for p in (WS / "helpers").glob("*b0712*"):
        if p.is_file():
            protected[str(p)] = sha(p)
    logs = WS / ("logs/ck-plugin-b0712-qualification-" + args.attempt)
    logs.mkdir()
    release = Path("/opt/bc250-stack/current").resolve(strict=True)
    manifest = json.loads((release / "runtime.json").read_text())
    groups = os.getgrouplist(pwd.getpwuid(manifest["uid"]).pw_name, manifest["gid"])
    groups += [Path(d).stat().st_gid for d in ("/dev/kfd", "/dev/dri/renderD128")]
    before = subprocess.check_output(["dmesg"])
    (logs / "kernel-before.log").write_bytes(before)
    report = dict(
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
            for number, folder in enumerate(a["roots"]):
                for profile in ("daily", "rccl-single"):
                    label = f"ck-plugin-b0712-{args.attempt}-extract{number}-{profile}"
                    scratch = base / label
                    scratch.mkdir()
                    os.chown(scratch, manifest["uid"], manifest["gid"])
                    # Make immutable-payload identity fixtures on the host filesystem.
                    # Container bind mounts prohibit creating these hardlinks there.
                    fixtures = scratch / "identity-fixtures"
                    fixtures.mkdir()
                    for kind in (
                        "native",
                        "architecture",
                        "payload",
                        "escape",
                        "extra",
                        "symlink",
                    ):
                        dest = fixtures / ("bad-" + kind)

                        def copy_fixture(source, target):
                            return (
                                shutil.copy2(source, target)
                                if Path(source).name == "manifest.json"
                                else os.link(source, target)
                            )

                        shutil.copytree(folder, dest, copy_function=copy_fixture)
                        for item in (dest, *dest.rglob("*")):
                            if item.is_dir() or item == dest / "manifest.json":
                                os.chown(item, manifest["uid"], manifest["gid"])
                        (dest / "manifest.json").chmod(0o644)
                    target = WS / "logs" / label
                    target.mkdir()
                    os.chown(target, manifest["uid"], manifest["gid"])
                    env = dict(
                        TMPDIR=str(scratch),
                        TMP=str(scratch),
                        TEMP=str(scratch),
                        PYTHONDONTWRITEBYTECODE="1",
                        BC250_CK_TEST_COHORT="daily" if profile == "daily" else "rccl",
                        MIOPEN_USER_DB_PATH=str(scratch / "miopen-db"),
                        MIOPEN_CUSTOM_CACHE_DIR=str(scratch / "miopen-cache"),
                    )
                    envfile = scratch / "environment"
                    envfile.write_text(
                        "".join(k + "=" + v + "\n" for k, v in env.items())
                    )
                    options = LaunchOptions(
                        arguments=[
                            str(WS / "helpers/validate_ck_plugin_b0712.py"),
                            "--artifact",
                            folder,
                        ],
                        mounts=[
                            (folder, "ro"),
                            (str(WS / "helpers"), "ro"),
                            (str(target), "rw"),
                        ],
                        environment=env,
                        runtime_profile=profile,
                    )
                    # Reuse installed launch context but replace automatic home/cwd mounts
                    # with this owned scratch directory. Original candidates are absent.
                    profile_release = selected_release(options)
                    profile_manifest = json.loads(
                        (profile_release / "runtime.json").read_text()
                    )
                    command = build_command(
                        profile_release,
                        profile_manifest,
                        scratch,
                        scratch,
                        options,
                        groups,
                        envfile,
                    )
                    target.rmdir()
                    report["gates"].append(gate(label, command, 9, manifest))
                    atomic_json(logs / "progress.json", report)
            for name, launcher, script, extra, count, profile in [
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
                label = "ck-plugin-b0712-" + args.attempt + "-" + name
                scratch = base / label
                scratch.mkdir()
                os.chown(scratch, manifest["uid"], manifest["gid"])
                cmd = ["/usr/local/bin/" + launcher, "--runtime-profile", profile]
                for k, v in dict(
                    **(
                        dict(BC250_CK_MARKER="installed", MIOPEN_LOG_LEVEL="6")
                        if name.startswith("installed-")
                        else {}
                    ),
                    TMPDIR=str(scratch),
                    TMP=str(scratch),
                    TEMP=str(scratch),
                    MIOPEN_USER_DB_PATH=str(scratch / "db"),
                    MIOPEN_CUSTOM_CACHE_DIR=str(scratch / "cache"),
                ).items():
                    cmd += ["--env", k + "=" + v]
                cmd += [str(WS / "helpers" / script), *extra]
                report["gates"].append(gate(label, cmd, count, manifest))
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
            links_unchanged=all(os.readlink(p) == v for p, v in a["links"].items()),
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
        report["status"] == "passed" and sum(g["checks"] for g in report["gates"]) == 78
    )
    print("CK_PLUGIN_B0712_QUALIFICATION_PASS", flush=True)


if __name__ == "__main__":
    main()
