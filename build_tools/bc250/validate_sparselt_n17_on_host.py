#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""N1.7 sparse Lt INT8 baseline/build/GPU gate; run under update_dev_prefix.py.

Usage on .104 (through doas): helpers/update_dev_prefix.py -- python3
helpers/validate_sparselt_n17_on_host.py WORKSPACE SOURCE_DELTA_TAR
The delta contains the sparse scalar implementation, rooted at source/rocm-libraries.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("source_delta", type=Path)
    parser.add_argument("--attempt", default="n17-sparselt")
    args = parser.parse_args()
    ws = args.workspace.resolve()
    logs = ws / "logs" / args.attempt
    logs.mkdir()
    report = {"task": "N1.7", "status": "running", "checks": []}
    output = logs / "result.json"
    prefix = "/workspace/build/dist/rocm"
    image = "localhost/bc250-rocm-build:10.2-ccache"
    library = ws / "build/dist/rocm/lib/libhipsparselt.so.0.2"
    backup = ws / "recovery/n17-sparselt-before.so.0.2"
    source = (
        ws
        / "source/rocm-libraries/projects/hipsparselt/library/src/hcc_detail/gfx1013_scalar.hip"
    )
    report["before_library_sha256"] = digest(library)
    report["source_delta_sha256"] = digest(args.source_delta)
    report["test_source_sha256"] = digest(ws / "helpers/hipsparselt_integer_smoke.cpp")
    report["resources"] = {"jobs": 16, "ccache": "20G", "sdma": False}

    def save() -> None:
        output.write_text(json.dumps(report, indent=2) + "\n")

    def run(label: str, command: list[str], timeout: int, expected: int = 0) -> str:
        report["phase"] = label
        save()
        start = time.monotonic()
        with (logs / f"{label}.log").open("w") as stream:
            result = subprocess.run(
                command, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout + 30
            )
        text = (logs / f"{label}.log").read_text()
        report["checks"].append(
            {
                "name": label,
                "exit": result.returncode,
                "seconds": time.monotonic() - start,
            }
        )
        save()
        if result.returncode != expected:
            raise RuntimeError(f"{label}: exit {result.returncode}; see {logs}")
        return text

    def container(gpu: bool, timeout: int, command: list[str]) -> list[str]:
        result = [
            "podman",
            "run",
            "--rm",
            "--name",
            "bc250-n17-sparselt",
            "--network",
            "none",
            "-v",
            f"{ws}:/workspace:rw",
            "-e",
            "HSA_ENABLE_SDMA=0",
            "-e",
            "OMP_NUM_THREADS=4",
            "-e",
            f"LD_LIBRARY_PATH={prefix}/lib:{prefix}/lib/llvm/lib",
            "-e",
            f"ROCM_PATH={prefix}",
            "-e",
            f"HIP_PATH={prefix}",
        ]
        if gpu:
            result.extend(["--device", "/dev/kfd", "--device", "/dev/dri/renderD128"])
        # Build inputs remain read-only even though build/cache/log mounts are writable.
        result.extend(
            [
                "-v",
                f"{ws / 'source'}:/workspace/source:ro",
                "-v",
                f"{ws / 'helpers'}:/workspace/helpers:ro",
                image,
                "timeout",
                "--kill-after=10s",
                f"{timeout}s",
            ]
        )
        return result + command

    changed = False
    save()
    try:
        assert (
            Path("/sys/class/watchdog/watchdog0/state").read_text().strip() == "active"
        )
        subprocess.run(
            ["dmesg"], stdout=(logs / "kernel-before.log").open("w"), check=True
        )
        backup.parent.mkdir(exist_ok=True)
        if backup.exists():
            assert (
                digest(backup) == report["before_library_sha256"]
            ), "Preserve prior backup"
        else:
            shutil.copy2(
                library, backup
            )  # independent copy: do not retain a dist hardlink
        shutil.copy2(source, logs / "sparselt-before.hip")
        binary = "/workspace/build/bc250-coverage/hipsparselt_n17_smoke"
        compile_command = [
            f"{prefix}/bin/hipcc",
            "-O2",
            "-std=c++17",
            "--offload-arch=gfx1013",
            f"-I{prefix}/include",
            f"-L{prefix}/lib",
            "/workspace/helpers/hipsparselt_integer_smoke.cpp",
            "-lhipsparselt",
            "-o",
            binary,
        ]
        run("compile-probe", container(False, 180, compile_command), 180)
        baseline = run(
            "baseline-int8", container(True, 90, [binary, "--baseline"]), 90
        )
        assert "BASELINE_REJECTED_INT8" in baseline
        members = subprocess.check_output(
            ["tar", "-tf", str(args.source_delta)], text=True
        ).splitlines()
        assert members == [
            "rocm-libraries/projects/hipsparselt/library/src/hcc_detail/gfx1013_scalar.hip"
        ], members
        subprocess.run(
            ["tar", "-xf", str(args.source_delta), "-C", str(ws / "source")], check=True
        )
        # A failed GPU gate restores the prior unified library; force the two
        # objects/staging steps on a retry so cached stamps cannot retain it.
        for member in members:
            (ws / "source" / member).touch()
        report["after_source_sha256"] = digest(source)
        changed = True
        run(
            "targeted-build",
            container(False, 600, ["bash", "/workspace/helpers/build_sparselt_n14.sh"]),
            600,
        )
        report["after_library_sha256"] = digest(library)
        assert report["before_library_sha256"] != report["after_library_sha256"]
        result = run("gpu-correctness", container(True, 120, [binary]), 120)
        match = re.search(r"HIPSPARSELT_INTEGER_PASS cases=(\d+)", result)
        assert match and int(match[1]) == len(re.findall(r"^PASS ", result, re.M))
        report["executed_cases"] = int(match[1])
        report["timings"] = [
            dict(zip(("case", "dtype", "output", "repeats", "mean_us"), row))
            for row in re.findall(
                r"^TIMING (\S+) dtype=(\d+) output=(\d+) repeats=(\d+) mean_us=(\S+)",
                result,
                re.M,
            )
        ]
        # Arithmetic/pruning paths changed: retain the already-qualified padded
        # batching/type/activation gate as a focused regression.
        old = run(
            "scalar-regression",
            container(
                True, 120, ["/workspace/build/bc250-coverage/hipsparselt_scalar_smoke"]
            ),
            120,
        )
        assert "HIPSPARSELT_SCALAR_PASS cases=234" in old
        report["scalar_regression_cases"] = 234
        broadcast = run(
            "broadcast-regression",
            container(
                True, 120, ["/workspace/build/bc250-coverage/hipsparselt_n14_smoke"]
            ),
            120,
        )
        assert "HIPSPARSELT_BROADCAST_PASS cases=609" in broadcast
        report["broadcast_regression_cases"] = 609
        bias = run(
            "bias-regression",
            container(
                True, 120, ["/workspace/build/bc250-coverage/hipsparselt_n15_smoke"]
            ),
            120,
        )
        assert "HIPSPARSELT_BIAS_PASS cases=567" in bias
        report["bias_regression_cases"] = 567
        vector = run("vector-regression", container(True, 120,
                     ["/workspace/build/bc250-coverage/hipsparselt_n16_smoke"]), 120)
        assert "HIPSPARSELT_VECTOR_PASS cases=438" in vector
        report["vector_regression_cases"] = 438
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        if changed:
            # Recovery runs before the exclusive runtime lock is released.
            shutil.copy2(backup, library)
            report["prefix_restored"] = (
                digest(library) == report["before_library_sha256"]
            )
        raise
    finally:
        subprocess.run(
            ["dmesg"], stdout=(logs / "kernel-after.log").open("w"), check=True
        )
        save()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
