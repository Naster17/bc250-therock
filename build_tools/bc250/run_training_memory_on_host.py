#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run the bounded training/memory milestone and stop before performance work."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse previously passed cases after an unrelated case fix",
    )
    args = parser.parse_args()
    if os.geteuid() != 0:
        raise RuntimeError("Run host orchestration as root")
    workspace = args.workspace.resolve()
    assert Path("/sys/class/watchdog/watchdog0/state").read_text().strip() == "active"
    logs = workspace / "logs/training-memory"
    logs.mkdir(exist_ok=True)
    # Container processes run as nik; results must be writable under the bind mount.
    ownership = workspace.stat()
    os.chown(logs, ownership.st_uid, ownership.st_gid)
    status = workspace / "logs/training-memory.exit"
    status.write_text("running\n")
    output = logs / "result.json"
    previous_checks = (
        json.loads(output.read_text())["checks"]
        if args.resume and output.exists()
        else []
    )
    report = {
        "status": "running",
        "checks": previous_checks.copy(),
        "reused_checks": previous_checks,
        "performance_started": False,
    }
    started = time.monotonic()
    with (logs / "kernel-before.log").open("w") as stream:
        subprocess.run(["dmesg"], stdout=stream, check=True)

    def passed(name: str) -> None:
        report["checks"].append(name)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"TRAINING_MEMORY_CASE_PASS {name}", flush=True)

    def run(name: str, arguments: list[str], timeout: int = 180) -> None:
        if name in previous_checks:
            print(f"TRAINING_MEMORY_REUSED_PASS {name}", flush=True)
            return
        with (logs / f"{name}.log").open("w") as stream:
            with subprocess.Popen(
                ["bc250-torch", *arguments],
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
            ) as process:
                try:
                    code = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    raise
                if code != 0:
                    raise RuntimeError(
                        f"Case {name} failed: {code}; see {logs / (name + '.log')}"
                    )
        passed(name)

    try:
        deadline = time.monotonic() + 300
        while not (workspace / "logs/training-model.exit").exists():
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Model preparation did not complete within the bound"
                )
            time.sleep(2)
        assert (workspace / "logs/training-model.exit").read_text().strip() == "0"
        environment = workspace / "training-env-stable/training.env"
        training = workspace / "training-validation"
        # Let the runtime user create the model/adapters below an owned directory.
        training.mkdir(exist_ok=True)
        os.chown(training, ownership.st_uid, ownership.st_gid)
        for mode in ("full", "lora-fp16", "lora-bf16"):
            for phase in ("train", "resume"):
                run(
                    f"{mode}-{phase}",
                    [
                        "--env-file",
                        str(environment),
                        "--env",
                        "HF_HUB_OFFLINE=1",
                        str(workspace / "helpers/validate_model_training.py"),
                        "--model",
                        str(workspace / "training-model"),
                        "--directory",
                        str(training),
                        "--mode",
                        mode,
                        "--phase",
                        phase,
                        "--steps",
                        "4",
                    ],
                )
        script = workspace / "helpers/validate_memory_concurrency.py"
        for mode in ("pressure", "streams"):
            run(
                mode,
                [str(script), "--mode", mode, "--output", str(logs / f"{mode}.json")],
                timeout=120,
            )
        processes = []
        streams = []
        try:
            for worker in (0, 1):
                stream = (logs / f"worker-{worker}.log").open("w")
                streams.append(stream)
                processes.append(
                    subprocess.Popen(
                        [
                            "bc250-torch",
                            str(script),
                            "--mode",
                            "worker",
                            "--worker",
                            str(worker),
                            "--output",
                            str(logs / f"worker-{worker}.json"),
                        ],
                        stdin=subprocess.DEVNULL,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                    )
                )
            deadline = time.monotonic() + 120
            for process in processes:
                code = process.wait(timeout=max(1, deadline - time.monotonic()))
                if code != 0:
                    raise RuntimeError(f"Concurrent runtime process failed: {code}")
            passed("two_concurrent_native_gpu_processes")
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
            for stream in streams:
                stream.close()
        report["status"] = "passed"
        status.write_text("0\n")
    except BaseException:
        report["status"] = "failed"
        status.write_text("1\n")
        raise
    finally:
        with (logs / "kernel-after.log").open("w") as stream:
            subprocess.run(["dmesg"], stdout=stream, check=True)
        report["kernel_log_unchanged"] = (logs / "kernel-before.log").read_bytes() == (
            logs / "kernel-after.log"
        ).read_bytes()
        report["seconds"] = time.monotonic() - started
        output.write_text(json.dumps(report, indent=2) + "\n")
    print("TRAINING_AND_MEMORY_MILESTONE_PASS_STOPPED_BEFORE_PERFORMANCE", flush=True)


if __name__ == "__main__":
    main()
