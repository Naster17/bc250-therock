#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run D55's full isolated clean Torch rebuild in the retained native builder."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from board_preflight import reservation
from build_sdk_d48_on_host import available

WORK = Path("/home/nik/bc250work")
WS = WORK / "rocm-10.2-gfx1013"
IMAGE = "28ad36e62398875f0ad77112b1d0f2b1cd0b3b55560d9cc38c6908cada8caa06"
SOURCE_SHA = "6867ed719c201cf7bc368800aa64c5e16a2e3a4c3c03485ad502684e2feb5723"
SIX_SHA = "70e8a77beed4562e7f14fe23a786b54f6296e34344c23bc42f07b15018ff98e9"
GIB = 1024**3
DEADLINE = 3600


def mount(source, target, mode="ro"):
    return ["-v", f"{source}:{target}:{mode}"]


def recipe_env_prefix(recipe):
    lines = recipe.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith("python -m pip install "):
            return "".join(lines[:index])
    raise AssertionError("Recipe has no pip boundary")


def run(profile, attempt):
    tools = Path(__file__).resolve().parent
    assert os.geteuid() == 0 and os.uname().nodename == "bc250"
    name = f"bc250-d55-{profile}-full-{attempt}"
    report = WORK / "evidence/source-d55" / name
    scratch = WS / "tmp" / name
    archive = WORK / "tools/source-d52/package-r1/torch-source-d52-r1.tar.gz"
    sdk = WS / "build/dist/rocm"
    rccl = WS / "rccl-b034/install-r1"
    result = dict(status="started", profile=profile, phase="full")
    with reservation(name) as preflight:
        report.mkdir(parents=True)
        scratch.mkdir()
        result["preflight"] = preflight
        child = None
        start = time.monotonic()
        try:
            assert available() >= 12 * GIB
            assert shutil.disk_usage(scratch).free >= 20 * GIB
            # The editable install writes into the image venv, so the
            # container layer stays writable; all preserved inputs are RO.
            command = [
                "podman",
                "run",
                "--rm",
                "--pull=never",
                "--network=none",
                "--name",
                name,
                "--label",
                f"bc250.d55={name}",
                "--cpus=4",
                "--cpuset-cpus=0-3",
                "--memory=10g",
                "--memory-swap=10g",
                "--pids-limit=2048",
                "--tmpfs=/tmp:rw,size=2g",
                "--entrypoint",
                "/bin/bash",
            ]
            command += mount(sdk, "/workspace/build/dist/rocm")
            command += mount(rccl, "/workspace/rccl")
            command += mount(report, "/evidence", "rw")
            command += mount(tools, "/probe")
            with archive.open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == SOURCE_SHA
            subprocess.run(
                ["tar", "-xzf", str(archive), "-C", str(scratch)],
                check=True,
                timeout=180,
            )
            kit = scratch / "torch-source-d52-r1"
            recipe = (kit / "recipes" / f"build-fresh-{profile}.sh").read_text()
            assert recipe_env_prefix(recipe)
            six = tools / "six-1.11.0.tar.gz"
            with six.open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == SIX_SHA
            offline = scratch / "offline"
            offline.mkdir()
            subprocess.run(
                ["tar", "-xzf", str(six), "-C", str(offline)],
                check=True,
                timeout=10,
            )
            command += mount(offline, "/workspace/offline")
            config = scratch / "cache-config"
            config.mkdir()
            command += mount(kit / "torch", "/workspace/pytorch-work", "rw")
            command += mount(kit / "recipes", "/workspace/recipes")
            for relative in ("setup_ccache.py", "posix_ccache_compiler_check.py"):
                command += mount(
                    WS / "source/build_tools" / relative,
                    f"/workspace/source/build_tools/{relative}",
                )
            command += mount(
                WS / "helpers/torch_toolchain.cmake",
                "/workspace/helpers/torch_toolchain.cmake",
            )
            command += mount(WS / "cache/data", "/workspace/cache/data")
            command += mount(config, "/workspace/cache/torch-config", "rw")
            command += [IMAGE, "/probe/full_build_d55.sh", profile]
            result["command"] = command
            result["minimum_available_bytes"] = available()
            result["minimum_free_bytes"] = shutil.disk_usage(scratch).free
            with (report / "probe.log").open("w") as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                result["pid"] = child.pid
                (report / "running.json").write_text(
                    json.dumps(result, indent=2) + "\n"
                )
                while child.poll() is None:
                    mem = available()
                    disk = shutil.disk_usage(scratch).free
                    result["minimum_available_bytes"] = min(
                        result["minimum_available_bytes"], mem
                    )
                    result["minimum_free_bytes"] = min(
                        result["minimum_free_bytes"], disk
                    )
                    assert mem >= 3 * GIB and disk >= 5 * GIB, (mem, disk)
                    assert time.monotonic() - start < DEADLINE, "Build deadline"
                    time.sleep(0.2)
                result["returncode"] = child.returncode
                assert child.returncode == 0, child.returncode
            assert (report / "full-build-passed").is_file()
            result["status"] = "passed"
        except BaseException as error:
            result.update(status="failed", error=repr(error))
            raise
        finally:
            if child and child.poll() is None:
                subprocess.run(
                    ["podman", "stop", "--time", "5", name], check=False, timeout=20
                )
                child.wait(timeout=20)
            shutil.rmtree(scratch)
            (report / "running.json").unlink(missing_ok=True)
            result["seconds"] = round(time.monotonic() - start, 3)
            (report / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("daily", "rccl-single"))
    parser.add_argument("attempt", type=int)
    args = parser.parse_args()
    run(args.profile, args.attempt)
