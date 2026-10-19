#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Targeted rocBLAS loader repair build; active prefix is mounted read-only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from deploy_torch_a04_on_host import atomic_json, sha

WS = Path("/home/nik/bc250work/rocm-10.2-gfx1013")
HEADER = "Tensile/Source/lib/include/Tensile/MasterSolutionLibrary.hpp"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", choices=("r1", "r2", "r3"), default="r1")
    args = parser.parse_args()
    assert os.geteuid() == 0
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    baseline = json.loads(
        (WS / "logs/training-n31-qualification-r4/final.json").read_text()
    )
    protected = baseline["protected_sha256"]
    for p, d in protected.items():
        assert sha(Path(p)) == d, p
    logs = WS / ("logs/gemm-n32-build-" + args.attempt)
    logs.mkdir()
    candidate = WS / ("candidates/gemm-n32-" + args.attempt)
    candidate.mkdir(parents=True)
    recovery = WS / "recovery/gemm-n32-before"
    staged = WS / "helpers/MasterSolutionLibrary.gemm-n32.hpp"
    files = [
        (
            WS / "source/rocm-libraries/shared/tensile" / HEADER,
            recovery / "MasterSolutionLibrary.source.hpp",
        ),
        (
            WS / "build/tensile-source" / HEADER,
            recovery / "MasterSolutionLibrary.derived.hpp",
        ),
    ]
    for target, backup in files:
        allowed = {sha(backup)}
        prior = WS / "logs/gemm-n32-build-r1/before.json"
        if prior.exists():
            allowed.add(json.loads(prior.read_text())["source_header_sha256"])
        assert sha(target) in allowed
        target.write_bytes(staged.read_bytes())
    report = dict(
        status="building",
        protected_sha256=protected,
        source_header_sha256=sha(staged),
        kernel_before=subprocess.check_output(["dmesg"]).decode(),
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        candidate=str(candidate),
        prefix_changed=False,
    )
    command = [
        "podman",
        "run",
        "--rm",
        "--name",
        "bc250-gemm-n32-build-" + args.attempt,
        "--network",
        "none",
    ]
    for src, dst, mode in (
        (WS / "source", "/workspace/source", "ro"),
        (WS / "helpers", "/workspace/helpers", "ro"),
        (WS / "build", "/workspace/build", "rw"),
        (WS / "cache", "/workspace/cache", "rw"),
        (WS / "build/dist/rocm", "/workspace/build/dist/rocm", "ro"),
        (candidate, "/workspace/candidate", "rw"),
    ):
        command += ["-v", f"{src}:{dst}:{mode}"]
    command += [
        "localhost/bc250-rocm-build:10.2-ccache",
        "bash",
        "/workspace/helpers/build_rocblas_gemm_n32.sh",
    ]
    report["command"] = command
    atomic_json(logs / "before.json", report)
    try:
        with (logs / "build.log").open("w") as log:
            subprocess.run(
                command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=1800
            )
        report.update(
            status="passed", candidate_sha256=sha(candidate / "librocblas.so.5.8")
        )
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        report["changed_hashes"] = [
            p for p, d in protected.items() if sha(Path(p)) != d
        ]
        report["kernel_log_unchanged"] = (
            subprocess.check_output(["dmesg"]).decode() == report["kernel_before"]
        )
        atomic_json(logs / "final.json", report)
    assert not report["changed_hashes"] and report["kernel_log_unchanged"]


if __name__ == "__main__":
    main()
