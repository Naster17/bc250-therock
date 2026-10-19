#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Profile the installed N3.2 stack while preserving its qualified identities."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from deploy_torch_a04_on_host import atomic_json, check_gate, sha

WS = Path("/home/nik/bc250work/rocm-10.2-gfx1013")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", default="r1")
    parser.add_argument("--length", type=int, choices=(65, 1024), default=1024)
    parser.add_argument("--phase", choices=("profile", "gemm"), default="profile")
    parser.add_argument(
        "--reuse-attempt", help="Seal existing passed GPU gates without rerunning them"
    )
    args = parser.parse_args()
    if args.reuse_attempt:
        assert args.phase == "gemm" and re.fullmatch(
            r"r[1-9][0-9]*", args.reuse_attempt
        )
    assert re.fullmatch(r"r[1-9][0-9]*", args.attempt)
    assert os.geteuid() == 0
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    assert os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize > 2 * 1024**3
    logs = WS / ("logs/training-n33-qualification-" + args.attempt)
    logs.mkdir()
    baseline = json.loads((WS / "logs/gemm-n32-deployment-r1/final.json").read_text())
    protected = dict(baseline["protected_sha256"])
    for path in (
        Path("/home/nik/finetune-n7/replica/train.py"),
        Path("/home/nik/finetune-n7/replica/bc250_ops.py"),
        Path("/home/nik/finetune-n7/scripts/train_bc250.py"),
        Path("/home/nik/finetune-n7/configs/qwen35-2b-lora.json"),
    ):
        protected.setdefault(str(path), sha(path))
    for p, digest in protected.items():
        assert sha(Path(p)) == digest, p
    source = {
        name: sha(WS / "helpers" / name)
        for name in (
            Path(__file__).name,
            "profile_training_n31.py",
            "profile_training_n33.py",
            "analyze_training_n31.py",
            "audit_gemm_n32.py",
            "benchmark_gemm_n32.py",
            "validate_qwen35_training_model.py",
            "validate_torch_a04_on_host.py",
            "update_dev_prefix.py",
        )
    }
    if args.reuse_attempt:
        original = json.loads(
            (
                WS
                / ("logs/training-n33-qualification-" + args.reuse_attempt)
                / "before.json"
            ).read_text()
        )
        for name, digest in original["source_sha256"].items():
            if name != Path(__file__).name:
                assert source[name] == digest, name
        assert original["protected_sha256"] == protected
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    before = subprocess.check_output(["dmesg"])
    (logs / "kernel-before.log").write_bytes(before)
    report = dict(
        status="running",
        protected_sha256=protected,
        source_sha256=source,
        links=baseline["links"],
        boot_id=boot,
        gates=[],
        torch_target=baseline["torch_target"],
        rocm_target=baseline["rocm_target"],
        runtime_sha256=baseline["runtime_sha256"],
        aot_caches=baseline["aot_caches"],
        sequence_length=args.length,
        phase=args.phase,
        reused_gpu_attempt=args.reuse_attempt,
        prefix_changed=False,
        library_or_torch_build_required=False,
    )
    atomic_json(logs / "before.json", report)
    try:
        groups = (
            [("profile", "profile_training_n33.py", ["--length", str(args.length)], 9)]
            if args.phase == "profile"
            else [
                ("layout", "audit_gemm_n32.py", ["--shape", "1024", "2048", "6144"], 2),
                (
                    "solutions",
                    "benchmark_gemm_n32.py",
                    [
                        "--audit-label",
                        "training-n33-"
                        + (args.reuse_attempt or args.attempt)
                        + "-layout",
                    ],
                    None,
                ),
            ]
        )
        for group, script, arguments, count in groups:
            label = "training-n33-" + (args.reuse_attempt or args.attempt) + "-" + group
            if group == "solutions":
                layout = (
                    WS
                    / "logs"
                    / (
                        "training-n33-"
                        + (args.reuse_attempt or args.attempt)
                        + "-layout"
                    )
                )
                count = len(json.loads((layout / "result.json").read_text())["layouts"])
                assert count > 0
            command = [
                sys.executable,
                str(WS / "helpers/validate_torch_a04_on_host.py"),
                "--guard-release",
                os.uname().release,
                "--label",
                label,
                "--script",
                script,
            ]
            command += ["--test-arg=" + a for a in arguments]
            if not args.reuse_attempt:
                with (logs / (label + ".log")).open("w") as log:
                    subprocess.run(
                        command,
                        check=True,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=570,
                    )
            check_gate(label)
            result = json.loads((WS / "logs" / label / "result.json").read_text())
            assert result["status"] == "passed" and len(result["checks"]) == count
            guard = json.loads((Path("/run") / label / "state.json").read_text())
            assert guard["status"] == "acknowledged" and guard["timeout_seconds"] == 600
            report["gates"].append(
                dict(label=label, checks=count, command=command, guard=guard)
            )
            atomic_json(logs / "progress.json", report)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        report["changed_hashes"] = [
            p for p, digest in protected.items() if sha(Path(p)) != digest
        ]
        report["source_hashes_unchanged"] = all(
            sha(WS / "helpers" / n) == d for n, d in source.items()
        )
        report["links_unchanged"] = all(
            os.readlink(p) == target for p, target in report["links"].items()
        )
        report["kernel_log_unchanged_entire_batch"] = (
            subprocess.check_output(["dmesg"]) == before
        )
        report["kernel"] = subprocess.check_output(["uname", "-r"], text=True).strip()
        report["disk_available_bytes"] = (
            os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize
        )
        report["running_containers"] = subprocess.check_output(
            ["podman", "ps", "-q"], text=True
        ).split()
        assert Path("/proc/sys/kernel/random/boot_id").read_text().strip() == boot
        if (
            report["changed_hashes"]
            or not report["source_hashes_unchanged"]
            or not report["links_unchanged"]
            or not report["kernel_log_unchanged_entire_batch"]
            or report["running_containers"]
        ):
            report["status"] = "failed"
        atomic_json(logs / "final.json", report)
    assert report["status"] == "passed"
    print("TRAINING_N33_PASS", logs)


if __name__ == "__main__":
    main()
