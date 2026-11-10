#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded CPU-only retained native build-input audit; no prefix changes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from build_inputs_d51 import sha, cache_values, build_settings, classify
from board_preflight import preflight

WORK = Path("/home/nik/bc250work")
WS = WORK / "rocm-10.2-gfx1013"
TOOLS = WORK / "tools/source-d51"
ROOT = WORK / "evidence/source-d51-r1"
BUILD_IMAGE = "28ad36e62398875f0ad77112b1d0f2b1cd0b3b55560d9cc38c6908cada8caa06"
RUNTIME_IMAGE = "1bb4ac0ed3766d9c70431c1d5119e03cd4d10e710f4c04cd54254b5b87cd45d5"


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def sample():
    available = (
        int(
            next(
                x.split()[1]
                for x in Path("/proc/meminfo").read_text().splitlines()
                if x.startswith("MemAvailable:")
            )
        )
        * 1024
    )
    return dict(available_bytes=available, disk_free_bytes=shutil.disk_usage(WORK).free)


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label", choices=["source-d51-r1", "source-d51-r2"], default="source-d51-r1"
    )
    args = parser.parse_args()
    ROOT = WORK / "evidence" / args.label
    assert os.geteuid() == 0 and os.uname().nodename == "bc250"
    ROOT.mkdir(exist_ok=False)
    record = dict(
        status="running",
        profiles={},
        images={},
        sources={p.name: sha(p) for p in TOOLS.glob("*.py")},
        samples=[],
    )
    start = time.monotonic()
    with (WORK / "evidence/.board-reservation.lock").open("a") as board, Path(
        "/opt/bc250-stack/current/.runtime.lock"
    ).open("r") as runtime:
        fcntl.flock(board, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(runtime, fcntl.LOCK_SH | fcntl.LOCK_NB)
        record["preflight"] = preflight("d51-audit")
        assert sample()["disk_free_bytes"] > 6 * 1024**3
        try:
            missing = json.loads(
                (WORK / "tools/source-d50/dependencies-r1/manifest.json").read_text()
            )["nodes"]
            missing = [
                dict(path=n["path"], pin=n["pin"])
                for n in missing
                if not n["available"]
            ]
            for profile, name in [("daily", "b052"), ("rccl-single", "b053")]:
                folder = ROOT / profile
                folder.mkdir()
                source = WS / ("pytorch-quant-" + name)
                for relative in [
                    "build/CMakeCache.txt",
                    "build/compile_commands.json",
                    "build/build.ninja",
                    "torch/version.py",
                ]:
                    shutil.copy2(source / relative, folder / Path(relative).name)
                recipe = WS / "helpers" / ("build_torch_" + name + ".sh")
                toolchain = WS / "helpers/torch_toolchain.cmake"
                shutil.copy2(recipe, folder / recipe.name)
                shutil.copy2(toolchain, folder / toolchain.name)
                flags = cache_values((folder / "CMakeCache.txt").read_text())
                classification = classify(
                    (folder / "CMakeCache.txt").read_text(),
                    json.loads((folder / "compile_commands.json").read_text()),
                    missing,
                )
                settings = build_settings(recipe.read_text())
                assert (
                    flags["USE_NCCL"]
                    == settings["USE_NCCL"]
                    == ("0" if profile == "daily" else "1")
                )
                old = json.loads(
                    (
                        WS / "logs" / ("quant-" + name + "-build") / "final.json"
                    ).read_text()
                )
                assert old["image"] == BUILD_IMAGE
                original = dict(
                    image=old["image"],
                    command=old["command"],
                    status=old["status"],
                    source_record_sha256=sha(
                        WS / "logs" / ("quant-" + name + "-build") / "final.json"
                    ),
                )
                write(folder / "retained-build.json", original)
                assert (folder / "version.py").read_text().find(
                    "git_version = 'Unknown'"
                ) >= 0
                record["profiles"][profile] = dict(
                    classification=classification,
                    settings=settings,
                    original=original,
                    files={p.name: sha(p) for p in folder.iterdir() if p.is_file()},
                    compiler_settings={
                        k: v
                        for k, v in flags.items()
                        if k
                        in [
                            "CMAKE_C_COMPILER",
                            "CMAKE_CXX_COMPILER",
                            "CMAKE_HIP_COMPILER",
                            "CMAKE_MAKE_PROGRAM",
                            "CMAKE_COMMAND",
                            "Python_EXECUTABLE",
                            "CMAKE_CXX_COMPILER_LAUNCHER",
                        ]
                    },
                    ninja_family_mentions={
                        s: (folder / "build.ninja").read_text().count(s)
                        for s in [
                            "third_party/aiter",
                            "third_party/flash-attention",
                            "third_party/mslk",
                        ]
                    },
                )
            for label, image in [
                ("native-build", BUILD_IMAGE),
                ("runtime", RUNTIME_IMAGE),
            ]:
                folder = ROOT / label
                folder.mkdir()
                os.chown(folder, 1000, 1000)
                actual = subprocess.check_output(
                    ["podman", "image", "inspect", image, "--format", "{{.Id}}"],
                    text=True,
                    timeout=10,
                ).strip()
                assert actual == image
                command = [
                    "podman",
                    "run",
                    "--name",
                    "bc250-d51-" + label,
                    "--label",
                    "bc250.d51=audit",
                    "--rm",
                    "--pull=never",
                    "--network=none",
                    "--read-only",
                    "--user",
                    "1000:1000",
                    "--cpus=2",
                    "--cpuset-cpus=0-3",
                    "--memory=1g",
                    "--memory-swap=1g",
                    "--pids-limit=128",
                    "--tmpfs",
                    "/tmp:rw,size=64m",
                    "-v",
                    str(WS / "build") + ":/workspace/build:ro",
                    "-v",
                    str(TOOLS) + ":/audit:ro",
                    "-v",
                    str(folder) + ":/output:rw",
                    image,
                    "/opt/torch-venv/bin/python",
                    "/audit/probe_build_image_d51.py",
                    "--output",
                    "/output/result.json",
                ]
                with (folder / "worker.log").open("w") as log:
                    child = subprocess.Popen(
                        command, stdout=log, stderr=subprocess.STDOUT
                    )
                    try:
                        childstart = time.monotonic()
                        while child.poll() is None:
                            point = sample()
                            record["samples"].append(point)
                            assert (
                                point["available_bytes"] >= 3 * 1024**3
                                and point["disk_free_bytes"] >= 5 * 1024**3
                            )
                            assert time.monotonic() - childstart < 120
                            time.sleep(0.1)
                        assert child.returncode == 0, (label, child.returncode)
                    finally:
                        ids = subprocess.check_output(
                            ["podman", "ps", "-q", "--filter", "label=bc250.d51=audit"],
                            text=True,
                        ).split()
                        for cid in ids:
                            subprocess.run(
                                ["podman", "stop", "--time", "5", cid],
                                check=True,
                                timeout=15,
                            )
                        if child.poll() is None:
                            child.terminate()
                        child.wait(timeout=15)
                result = json.loads((folder / "result.json").read_text())
                assert result["status"] == "passed"
                record["images"][label] = dict(
                    image=image, command=command, result=result
                )
            wheels = []
            for file in (WS / "wheels").glob("*.whl"):
                wheels.append(
                    dict(name=file.name, bytes=file.stat().st_size, sha256=sha(file))
                )
            record["retained_wheels"] = wheels
            record["sdk_sha256"] = sha(
                Path("/opt/bc250-stack/current/rocm/lib/librocprofiler-sdk.so.1.4.2")
            )
            assert (
                record["sdk_sha256"]
                == "2a975370ba4eb1c2fe39e406ff36b175f554a49d0aa19febbd326f641fcb590b"
            )
            record.update(
                status="passed",
                seconds=time.monotonic() - start,
                postflight=preflight("d51-final"),
            )
        except BaseException as error:
            record.update(status="failed", error=repr(error))
            raise
        finally:
            write(ROOT / "result.json", record)


if __name__ == "__main__":
    main()
