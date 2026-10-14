#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""N1.8 native hipBLASLt small/skinny baseline/build/GPU gate; run under update_dev_prefix.py.

Usage on .104 (through doas): helpers/update_dev_prefix.py -- python3
helpers/validate_lt_n18_on_host.py WORKSPACE SOURCE_DELTA_TAR
The delta contains the Lt scalar implementation, rooted at source/rocm-libraries.
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
    parser.add_argument("--attempt", default="n18-lt")
    args = parser.parse_args()
    ws = args.workspace.resolve()
    logs = ws / "logs" / args.attempt
    logs.mkdir()
    report = {"task": "N1.8", "status": "running", "checks": []}
    output = logs / "result.json"
    prefix = "/workspace/build/dist/rocm"
    image = "localhost/bc250-rocm-build:10.2-ccache"
    library = ws / "build/dist/rocm/lib/libhipblaslt.so.1.5"
    backup = ws / "recovery/n18-lt-before.so.1.5"
    source = (
        ws
        / "source/rocm-libraries/projects/hipblaslt/library/src/amd_detail/gfx1013_scalar.cpp"
    )
    report["before_library_sha256"] = digest(library)
    report["source_delta_sha256"] = digest(args.source_delta)
    report["test_source_sha256"] = digest(ws / "helpers/hipblaslt_skinny_smoke.cpp")
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
            "bc250-n18-lt",
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

    pack = ws / "build/dist/rocm/.kpack/blas_lib_gfx1013.kpack"
    pack_backup = ws / "recovery/n18-blas_lib_gfx1013-before.kpack"
    report["before_pack_sha256"] = digest(pack)
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
        shutil.copy2(source, logs / "lt-before.cpp")
        if pack_backup.exists():
            assert digest(pack_backup) == report["before_pack_sha256"]
        else:
            shutil.copy2(pack, pack_backup)
        binary = "/workspace/build/bc250-coverage/hipblaslt_n18_smoke"
        run("compile-probe", container(False, 180, [f"{prefix}/bin/hipcc",
            "-O2", "-std=c++17", "--offload-arch=gfx1013", f"-I{prefix}/include",
            f"-L{prefix}/lib", "/workspace/helpers/hipblaslt_skinny_smoke.cpp",
            "-lhipblaslt", "-o", binary]), 180)
        def probe(label: str, timing: bool = False) -> str:
            result = run(label, container(True, 120, [binary] + (["--timing"] if timing else [])), 120)
            count = re.search(r"HIPBLASLT_SKINNY_PASS cases=(\d+)", result)
            assert count and int(count[1]) == len(re.findall(r"^PASS ", result, re.M))
            report[label + "_cases"] = int(count[1])
            if timing:
                report[label + "_timings"] = [dict(zip(
                    ("m", "n", "k", "input", "output", "row", "ta", "tb", "batches", "sample", "us"), row))
                    for row in re.findall(r"^TIMING m=(\d+) n=(\d+) k=(\d+) input=(\d+) output=(\d+) row=(\d+) ta=(\d+) tb=(\d+) batches=(\d+) sample=(\d+) us=(\S+)", result, re.M)]
                assert len(report[label + "_timings"]) == int(count[1]) * 5
            save()
            return result
        probe("baseline-correctness")
        probe("baseline-A1", True)
        members = subprocess.check_output(["tar", "-tf", str(args.source_delta)], text=True).splitlines()
        assert members == ["rocm-libraries/projects/hipblaslt/library/src/amd_detail/gfx1013_scalar.cpp"], members
        subprocess.run(["tar", "-xf", str(args.source_delta), "-C", str(ws / "source")], check=True)
        for member in members:
            (ws / "source" / member).touch()
        report["after_source_sha256"] = digest(source)
        changed = True
        run("targeted-build", container(False, 600, ["bash", "/workspace/helpers/build_lt_n18.sh"]), 600)
        report["after_library_sha256"] = digest(library)
        report["after_pack_sha256"] = digest(pack)
        assert report["before_library_sha256"] != report["after_library_sha256"]
        probe("gpu-correctness")
        candidate = logs / "candidate-libhipblaslt.so.1.5"
        shutil.copy2(library, candidate)
        candidate_pack = logs / "candidate-blas_lib_gfx1013.kpack"
        shutil.copy2(pack, candidate_pack)
        probe("candidate-B1", True)
        probe("candidate-B2", True)
        shutil.copy2(backup, library)
        shutil.copy2(pack_backup, pack)
        assert digest(library) == report["before_library_sha256"]
        probe("baseline-A2", True)
        shutil.copy2(candidate, library)
        shutil.copy2(candidate_pack, pack)
        assert digest(library) == report["after_library_sha256"]
        for label, name, marker, count in [
            ("scalar-regression", "hipblaslt_scalar_smoke", "HIPBLASLT_SCALAR_PASS", 403),
            ("extended-regression", "hipblaslt_priority1_smoke", "HIPBLASLT_PRIORITY1_PASS", 561)]:
            result = run(label, container(True, 120, ["/workspace/build/bc250-coverage/" + name]), 120)
            assert f"{marker} cases={count}" in result
            report[label + "_cases"] = count
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        if changed:
            # Recovery runs before the exclusive runtime lock is released.
            shutil.copy2(backup, library)
            shutil.copy2(pack_backup, pack)
            report["pack_restored"] = digest(pack) == report["before_pack_sha256"]
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
