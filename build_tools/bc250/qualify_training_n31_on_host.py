#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Preserve qualified N2 artifacts while profiling the next N3 training batch."""

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
    args = parser.parse_args()
    assert re.fullmatch(r"r[1-9][0-9]*", args.attempt)
    assert os.geteuid() == 0
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    assert os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize > 2 * 1024**3
    logs = WS / ("logs/training-n31-qualification-" + args.attempt)
    logs.mkdir()
    baseline = json.loads(
        (WS / "logs/ck-dl-n218-qualification-r4/final.json").read_text()
    )
    protected = dict(baseline["protected_sha256"])
    protected.update(baseline["binary_sha256"])
    for path in (
        Path("/home/nik/finetune-n7/replica/train.py"),
        Path("/home/nik/finetune-n7/replica/bc250_ops.py"),
        Path("/home/nik/finetune-n7/scripts/train_bc250.py"),
        Path("/home/nik/finetune-n7/configs/qwen35-2b-lora.json"),
    ):
        protected[str(path)] = sha(path)
    for p, digest in protected.items():
        assert sha(Path(p)) == digest, p
    source = {
        name: sha(WS / "helpers" / name)
        for name in (
            Path(__file__).name,
            "profile_training_n31.py",
            "analyze_training_n31.py",
            "validate_qwen35_training_model.py",
            "validate_torch_a04_on_host.py",
            "update_dev_prefix.py",
        )
    }
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
        prefix_changed=False,
        library_or_torch_build_required=False,
    )
    atomic_json(logs / "before.json", report)
    try:
        for group, script, arguments, count in (
            ("profile", "profile_training_n31.py", ["--length", str(args.length)], 8),
        ):
            label = "training-n31-" + args.attempt + "-" + group
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
    print("TRAINING_N31_PASS", logs)


if __name__ == "__main__":
    main()
