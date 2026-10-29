#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Configure/build a matching isolated ORT core/Python/MIGraphX set, offline."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

from bc250_migraphx import registered
from bc250_torch import LaunchOptions, runtime_lock
from build_migraphx_b067_on_host import verify_hashes
from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight

BASE = WS / "tmp/onnxruntime-b0610"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attempt", required=True)
    p.add_argument("--task", help="reuse an existing task for an incremental retry")
    p.add_argument("--phase", choices=("configure", "build"), required=True)
    args = p.parse_args()
    assert args.attempt in ("r1", "r2", "r3", "r4", "r5")
    preflight("onnxruntime-b0610-" + args.phase + "-" + args.attempt)
    task_name = args.task or args.attempt
    assert task_name in ("r1", "r2", "r3", "r4", "r5")
    task = BASE / ("task-" + task_name)
    manifest = json.loads(Path("/opt/bc250-stack/current/runtime.json").read_text())
    baseline = json.loads(
        (WS / "logs/onnxruntime-b069-qualification-r2/final.json").read_text()
    )
    inputs = BASE / "incoming"
    plan = json.loads(
        (WS / "tmp/onnxruntime-b069/inputs/candidate-plan.json").read_text()
    )
    with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
        assert not verify_hashes(baseline["protected_sha256"])
        registration, sdk = registered(LaunchOptions())
        available = os.statvfs(BASE).f_bavail * os.statvfs(BASE).f_frsize
        assert available > (24 if args.phase == "configure" else 4) * 1024**3
        source = BASE / "source-r1"
        if not source.exists():
            archive = WS / "tmp/onnxruntime-b069/inputs/onnxruntime-source.tar.gz"
            audit = json.loads(
                (WS / "tmp/onnxruntime-b069/inputs/source-audit.json").read_text()
            )
            assert sha(archive) == audit["source_archive_sha256"]
            source.mkdir()
            with tarfile.open(archive) as t:
                t.extractall(source, filter="data")
            assert not verify_hashes(
                {str(source / n): d for n, d in audit["source_files_sha256"].items()}
            )
        mirror = BASE / "mirror"
        if not mirror.exists():
            with tarfile.open(inputs / "mirror.tar.gz") as t:
                t.extractall(BASE, filter="data")
        assert sha(mirror / "manifest.json") == sha(
            WS / "helpers/onnxruntime_b0610_dependencies.json"
        )
        dependencies = json.loads((mirror / "manifest.json").read_text())
        for spec in dependencies["dependencies"].values():
            assert sha(mirror / spec["relative_path"]) == spec["sha256"]
        private = BASE / "sdk-private-r1"
        if not private.exists():
            shutil.copytree(sdk, private, symlinks=True)
            header = WS / "helpers/migraphx-b069-cpp20.hpp"
            assert (
                sha(header)
                == "ca93e78779221bb1696460ec5d5eb14caec694b0d3431c275295e54a9a00454a"
            )
            shutil.copyfile(header, private / "include/migraphx/migraphx.hpp")
        if args.phase == "configure":
            if args.task:
                assert task.exists()
            else:
                task.mkdir()
        else:
            assert task.exists()
        logs = BASE / (args.phase + "-" + args.attempt)
        logs.mkdir()
        shutil.copyfile(Path(__file__), logs / "controller-source.py")
        script = WS / "helpers/build_onnxruntime_b0610.sh"
        shutil.copyfile(script, logs / "phase-script.sh")
        command = [
            "podman",
            "run",
            "--rm",
            "--name",
            "bc250-ort-b0610-" + args.phase + "-" + args.attempt,
            "--network=none",
            "--read-only",
            "--cpus=4",
            "--memory=10g",
            "--memory-swap=12g",
            "-v",
            str(source) + ":/source:ro",
            "-v",
            str(mirror) + ":/mirror:ro",
            "-v",
            str(private) + ":/sdk:ro",
            "-v",
            str(Path("/opt/bc250-stack/current/rocm").resolve())
            + ":/workspace/build/dist/rocm:ro",
            "-v",
            str(WS / "source/build_tools") + ":/workspace/source/build_tools:ro",
            "-v",
            str(task) + ":/task:rw",
            "-v",
            str(logs) + ":/logs:rw",
            "-v",
            str(logs / "phase-script.sh") + ":/phase.sh:ro",
            manifest["image"],
            "bash",
            "/phase.sh",
            args.phase,
        ]
        report = dict(
            status="running",
            phase=args.phase,
            attempt=args.attempt,
            command=command,
            started_unix=time.time(),
            ordinary_available_before=available,
            protected_sha256=baseline["protected_sha256"],
            links=baseline["links"],
            installed_registration=registration,
            ort_source="402979ac1bef89dcd55add41eaa7277c407599eb",
            installed_runtime_changed=False,
            host105_contacted=False,
        )
        atomic_json(logs / "progress.json", report)
        before = subprocess.check_output(["dmesg"])
        (logs / "kernel-before.log").write_bytes(before)
        with (logs / "controller.log").open("w") as stream:
            proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            report["controller_child_pid"] = proc.pid
            atomic_json(logs / "progress.json", report)
            reason = None
            try:
                while proc.poll() is None:
                    free = os.statvfs(BASE).f_bavail * os.statvfs(BASE).f_frsize
                    if (
                        free < 4 * 1024**3
                        or available - free > 20 * 1024**3
                        or time.time() - report["started_unix"] > 7200
                    ):
                        reason = "disk-or-deadline-bound"
                        subprocess.run(
                            [
                                "podman",
                                "stop",
                                "--time",
                                "15",
                                command[command.index("--name") + 1],
                            ],
                            check=True,
                        )
                        break
                    time.sleep(5)
                code = proc.wait(timeout=30)
            finally:
                if proc.poll() is None:
                    subprocess.run(
                        [
                            "podman",
                            "stop",
                            "--time",
                            "15",
                            command[command.index("--name") + 1],
                        ],
                        check=True,
                    )
                    proc.wait(timeout=30)
        report.update(
            status="passed" if code == 0 and reason is None else "failed",
            returncode=code,
            bound_failure=reason,
            ordinary_available_after=os.statvfs(BASE).f_bavail
            * os.statvfs(BASE).f_frsize,
            changed_hashes=verify_hashes(baseline["protected_sha256"]),
            kernel_log_unchanged=subprocess.check_output(["dmesg"]) == before,
        )
        if report["changed_hashes"] or not report["kernel_log_unchanged"]:
            report["status"] = "failed"
        (logs / "kernel-after.log").write_bytes(subprocess.check_output(["dmesg"]))
        atomic_json(logs / "final.json", report)
        assert report["status"] == "passed", str(logs / "final.json")
    print("ORT_B0610_" + args.phase.upper() + "_PASS", flush=True)


if __name__ == "__main__":
    main()
