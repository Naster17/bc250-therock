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


def main(
    *,
    batch_id: str = "training-n33",
    shape: tuple[int, int, int] = (1024, 2048, 6144),
    baseline_label: str = "gemm-n32-deployment-r1",
    default_phase: str = "profile",
    extra_gates: tuple = (),
    additional_sources: tuple[str, ...] = (),
    scratch_mounts: tuple[Path, ...] = (),
    minimum_disk_free_bytes: int = 2 * 1024**3,
    release_override: Path | None = None,
    hip_candidate: Path | None = None,
    include_training_app: bool = True,
) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", default="r1")
    parser.add_argument("--length", type=int, choices=(65, 1024), default=1024)
    parser.add_argument(
        "--phase", choices=("profile", "gemm", "extra"), default=default_phase
    )
    parser.add_argument(
        "--reuse-attempt", help="Seal existing passed GPU gates without rerunning them"
    )
    args = parser.parse_args()
    if args.reuse_attempt:
        assert args.phase == "gemm" and re.fullmatch(
            r"r[1-9][0-9]*", args.reuse_attempt
        )
    assert args.phase != "extra" or extra_gates
    assert re.fullmatch(r"r[1-9][0-9]*", args.attempt)
    assert re.fullmatch(
        r"(?:training-n[0-9]+|vision-b041|audio-b042|codec-b043|video-b044|image-b045|fft-b011|sparse-b021|distributed-b031|distributed-b032|rccl-b033|rccl-b034)",
        batch_id,
    )
    assert Path(baseline_label).name == baseline_label
    assert len(shape) == 3 and min(shape) > 0
    assert os.geteuid() == 0
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    assert os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize > minimum_disk_free_bytes
    logs = WS / ("logs/" + batch_id + "-qualification-" + args.attempt)
    logs.mkdir()
    baseline = json.loads((WS / "logs" / baseline_label / "final.json").read_text())
    protected = dict(baseline["protected_sha256"])
    app_paths = (
        Path("/home/nik/finetune-n7/replica/train.py"),
        Path("/home/nik/finetune-n7/replica/bc250_ops.py"),
        Path("/home/nik/finetune-n7/scripts/train_bc250.py"),
        Path("/home/nik/finetune-n7/configs/qwen35-2b-lora.json"),
    )
    for path in app_paths if include_training_app else ():
        protected.setdefault(str(path), sha(path))
    for p, digest in protected.items():
        assert sha(Path(p)) == digest, p
    source = {
        name: sha(WS / "helpers" / name)
        for name in (
            Path(__file__).name,
            Path(sys.argv[0]).name,
            "profile_training_n31.py",
            "profile_training_n33.py",
            "analyze_training_n31.py",
            "audit_gemm_n32.py",
            "benchmark_gemm_n32.py",
            "validate_qwen35_training_model.py",
            "validate_torch_a04_on_host.py",
            "update_dev_prefix.py",
            *additional_sources,
        )
    }
    if args.reuse_attempt:
        original = json.loads(
            (
                WS
                / ("logs/" + batch_id + "-qualification-" + args.reuse_attempt)
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
        audited_shape=list(shape),
        baseline_label=baseline_label,
        phase=args.phase,
        reused_gpu_attempt=args.reuse_attempt,
        hip_candidate=str(hip_candidate) if hip_candidate else None,
        hip_candidate_sha256=sha(hip_candidate) if hip_candidate else None,
        prefix_changed=False,
        library_or_torch_build_required=False,
        scratch_mounts=[str(p) for p in scratch_mounts],
        minimum_disk_free_bytes=minimum_disk_free_bytes,
        release_override=str(release_override) if release_override else None,
    )
    atomic_json(logs / "before.json", report)
    try:
        groups = (
            [("profile", "profile_training_n33.py", ["--length", str(args.length)], 9)]
            if args.phase == "profile"
            else (
                [
                    ("layout", "audit_gemm_n32.py", ["--shape", *map(str, shape)], 2),
                    (
                        "solutions",
                        "benchmark_gemm_n32.py",
                        [
                            "--audit-label",
                            batch_id
                            + "-"
                            + (args.reuse_attempt or args.attempt)
                            + "-layout",
                        ],
                        None,
                    ),
                ]
                if args.phase == "gemm"
                else extra_gates
            )
        )
        for group, script, arguments, count in groups:
            label = batch_id + "-" + (args.reuse_attempt or args.attempt) + "-" + group
            if group == "solutions":
                layout = (
                    WS
                    / "logs"
                    / (
                        batch_id
                        + "-"
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
            if hip_candidate is not None:
                command += ["--hip-candidate", str(hip_candidate)]
            if release_override is not None:
                command += ["--release", str(release_override)]
            for path in scratch_mounts:
                command += ["--mount-rw", str(path)]
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
        report["hip_candidate_unchanged"] = (
            not hip_candidate or sha(hip_candidate) == report["hip_candidate_sha256"]
        )
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
            or not report["hip_candidate_unchanged"]
            or not report["source_hashes_unchanged"]
            or not report["links_unchanged"]
            or not report["kernel_log_unchanged_entire_batch"]
            or report["running_containers"]
        ):
            report["status"] = "failed"
        atomic_json(logs / "final.json", report)
    assert report["status"] == "passed"
    print(batch_id.upper().replace("-", "_") + "_PASS", logs)


if __name__ == "__main__":
    main()
