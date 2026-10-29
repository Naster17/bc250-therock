#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Guard modern Torch exports and their public ORT GPU consumers on .104."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from bc250_ort import registered
from bc250_torch import LaunchOptions, runtime_lock
from build_migraphx_b067_on_host import verify_hashes
from deploy_torch_a04_on_host import atomic_json, sha
from qualify_rccl_b038_on_host import gate
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attempt", required=True)
    args = p.parse_args()
    assert args.attempt in ("r1", "r2", "r3", "r4")
    preflight("onnxruntime-b0612-" + args.attempt)
    base = WS / "tmp/onnxruntime-b0612"
    base.mkdir(exist_ok=True)
    logs = WS / ("logs/onnxruntime-b0612-qualification-" + args.attempt)
    logs.mkdir()
    baseline = json.loads(
        (WS / "logs/onnxruntime-b0611-acceptance-r2/final.json").read_text()
    )
    protected = dict(baseline["protected_sha256"])
    for path in (WS / "helpers").glob("*b0612*"):
        if path.is_file():
            protected[str(path)] = sha(path)
    before = subprocess.check_output(["dmesg"])
    (logs / "kernel-before.log").write_bytes(before)
    manifest = json.loads(Path("/opt/bc250-stack/current/runtime.json").read_text())
    owned = base / ("generation-" + args.attempt)
    owned.mkdir()
    os.chown(owned, manifest["uid"], manifest["gid"])
    fixtures = owned / "models"
    report = dict(
        status="running",
        gates=[],
        protected_sha256=protected,
        links=baseline["links"],
        aot_caches=baseline["aot_caches"],
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        fixtures=str(fixtures),
        runtime_changed=False,
        host105_contacted=False,
    )

    def command(launcher, label, script, extra, profile="daily"):
        scratch = base / label
        scratch.mkdir()
        os.chown(scratch, manifest["uid"], manifest["gid"])
        cmd = ["/usr/local/bin/" + launcher, "--runtime-profile", profile]
        for name, value in dict(
            TMPDIR=str(scratch),
            TMP=str(scratch),
            TEMP=str(scratch),
            MIOPEN_USER_DB_PATH=str(scratch / "miopen-db"),
            MIOPEN_CUSTOM_CACHE_DIR=str(scratch / "miopen-cache"),
        ).items():
            cmd += ["--env", name + "=" + value]
        return cmd + [str(WS / "helpers" / script), *extra]

    try:
        with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
            assert not verify_hashes(protected)
            registered(LaunchOptions())
            registered(LaunchOptions(runtime_profile="rccl-single"))
            label = "onnxruntime-b0612-" + args.attempt + "-generate"
            report["gates"].append(
                gate(
                    label,
                    command(
                        "bc250-torch",
                        label,
                        "generate_onnxruntime_b0612.py",
                        ["--fixtures", str(fixtures)],
                    ),
                    30,
                    manifest,
                )
            )
            generated = json.loads((fixtures / "manifest.json").read_text())
            assert generated["status"] == "passed" and len(generated["models"]) == 6
            for n, d in generated["files_sha256"].items():
                protected[str(fixtures / n)] = d
            protected[str(fixtures / "manifest.json")] = sha(fixtures / "manifest.json")
            atomic_json(logs / "progress.json", report)
            print("PASS", label, 30, flush=True)
            for profile in ("daily", "rccl-single"):
                label = "onnxruntime-b0612-" + args.attempt + "-" + profile
                report["gates"].append(
                    gate(
                        label,
                        command(
                            "bc250-ort",
                            label,
                            "validate_onnxruntime_b0612.py",
                            ["--fixtures", str(fixtures)],
                            profile,
                        ),
                        30,
                        manifest,
                    )
                )
                atomic_json(logs / "progress.json", report)
                print("PASS", label, 30, flush=True)
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
                    "daily-consumers",
                    "bc250-train",
                    "validate_rccl_b038.py",
                    ["--daily", "--consumers"],
                    5,
                ),
                (
                    "rccl-consumers",
                    "bc250-train",
                    "validate_rccl_b038.py",
                    ["--consumers"],
                    6,
                ),
            ]:
                label = "onnxruntime-b0612-" + args.attempt + "-" + name
                profile = "rccl-single" if name == "rccl-consumers" else "daily"
                report["gates"].append(
                    gate(
                        label,
                        command(launcher, label, script, extra, profile),
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
            links_unchanged=all(
                os.readlink(p) == v for p, v in report["links"].items()
            ),
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
        report["status"] == "passed"
        and sum(g["checks"] for g in report["gates"]) == 112
    )
    print("ORT_B0612_EXPORT_INTEGRATION_PASS", flush=True)


if __name__ == "__main__":
    main()
