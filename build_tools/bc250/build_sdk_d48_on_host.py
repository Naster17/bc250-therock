#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Resource-bounded CPU-only candidate build; retained sources/builds are read-only."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from board_preflight import reservation

WORK = Path("/home/nik/bc250work")
WS = WORK / "rocm-10.2-gfx1013"
TOOLS = WORK / "tools/sdk-d48"
ROOT = WORK / "evidence/sdk-d48-build-r2"
GIB = 1024**3


def available():
    return (
        int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
        )
        * 1024
    )


def owned():
    return subprocess.check_output(
        ["podman", "ps", "-q", "--filter", "label=bc250.d48=build-r2"],
        text=True,
        timeout=10,
    ).split()


def main():
    assert os.geteuid() == 0 and os.uname().nodename == "bc250"
    ROOT.mkdir(exist_ok=False)
    for name in (
        "agent.cpp",
        "sdk-recipe.json",
        "runtime-rpath.json",
        "build_sdk_d48_in_container.py",
        "audit_elf_d46.py",
    ):
        shutil.copy2(TOOLS / name, ROOT / name)
    image = json.loads(
        (WORK / "evidence/source-d47-r1/sdk-build-admission.json").read_text()
    )["image"]
    command = [
        "podman",
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--read-only",
        "--label=bc250.d48=build-r2",
        "--cpus=1",
        "--cpuset-cpus=0-3",
        "--memory=4g",
        "--memory-swap=4g",
        "--pids-limit=256",
        "--tmpfs=/tmp:rw,size=256m",
        "--shm-size=256m",
        "-v",
        f"{WS / 'build'}:/workspace/build:ro",
        "-v",
        f"{WS / 'source'}:/workspace/source:ro",
        "-v",
        f"{WS / 'cache'}:/workspace/cache:rw",
        "-v",
        f"{ROOT}:/workspace/d48:rw",
        "-e",
        "CCACHE_DIR=/workspace/cache/data",
        "-e",
        "CCACHE_CONFIGPATH=/workspace/cache/torch-config/ccache.conf",
        "-e",
        "CCACHE_MAXSIZE=20G",
        "-e",
        "LD_LIBRARY_PATH=/workspace/build/dist/rocm/lib:/workspace/build/dist/rocm/lib/llvm/lib:/workspace/build/dist/rocm/lib/rocm_sysdeps/lib",
        "--entrypoint=python3",
        image,
        "/workspace/d48/build_sdk_d48_in_container.py",
    ]
    result = dict(status="running", command=command, samples=[])
    with reservation("d48-build"):
        assert available() > 7 * GIB and shutil.disk_usage(ROOT).free > 9 * GIB
        try:
            start = time.monotonic()
            with (ROOT / "build.log").open("w") as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                try:
                    while child.poll() is None:
                        sample = dict(
                            available_bytes=available(),
                            disk_free_bytes=shutil.disk_usage(ROOT).free,
                        )
                        result["samples"].append(sample)
                        assert (
                            sample["available_bytes"] >= 3 * GIB
                            and sample["disk_free_bytes"] >= 5 * GIB
                        ), sample
                        assert time.monotonic() - start < 600, "Build deadline"
                        time.sleep(0.1)
                    assert child.returncode == 0, child.returncode
                finally:
                    for cid in owned():
                        subprocess.run(
                            ["podman", "stop", "--time", "5", cid],
                            check=True,
                            timeout=15,
                        )
                    child.wait(timeout=15)
            result.update(
                status="built-unqualified",
                seconds=time.monotonic() - start,
                candidate=json.loads((ROOT / "build.json").read_text()),
            )
        except BaseException as error:
            result.update(status="failed", error=repr(error))
            raise
        finally:
            assert not owned()
            (ROOT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], seconds=result["seconds"])))


if __name__ == "__main__":
    main()
