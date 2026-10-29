#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Explicit registered ORT/MIGraphX package activation, preserving ordinary defaults."""
import json
import os
from pathlib import Path
import sys

from bc250_migraphx import registered as sdk_registered, sha, trusted
from bc250_torch import (
    exit_status,
    parse_launcher_arguments,
    prepare_python_arguments,
    run_process,
    runtime_lock,
)

BASE = Path("/opt/bc250-stack/sdks")
NAME = "bc250-ort-migraphx-1.31.0-gfx1013-cp312-b0611"
REGISTRY = BASE / "ort-migraphx.json"


def check_package(root, record):
    trusted(root)
    info = root / "share/bc250-ort"
    if (
        sha(info / "sha256.json") != record["index_sha256"]
        or sha(info / "manifest.json") != record["manifest_sha256"]
    ):
        raise ValueError("ORT index/manifest identity mismatch")
    index = json.loads((info / "sha256.json").read_text())
    expected = (
        set(index["files"]) | set(index["symlinks"]) | {"share/bc250-ort/sha256.json"}
    )
    actual = set()
    for path in root.rglob("*"):
        name = str(path.relative_to(root))
        if path.is_symlink():
            if (
                name not in index["symlinks"]
                or os.readlink(path) != index["symlinks"][name]
                or not path.resolve(strict=True).is_relative_to(root)
            ):
                raise ValueError("ORT link identity mismatch: " + name)
            if path.lstat().st_uid != 0:
                raise ValueError("Untrusted ORT link: " + name)
            actual.add(name)
        else:
            trusted(path)
            if path.is_file():
                actual.add(name)
                if name in index["files"] and sha(path) != index["files"][name]:
                    raise ValueError("ORT file identity mismatch: " + name)
            elif not path.is_dir():
                raise ValueError("Unsupported ORT package entry")
    if actual != expected:
        raise ValueError("ORT package file set mismatch")
    return json.loads((info / "manifest.json").read_text())


def registered(options):
    trusted(REGISTRY)
    record = json.loads(REGISTRY.read_text())
    root = Path(record["package"])
    if root != BASE / NAME:
        raise ValueError("Unexpected ORT registration")
    manifest = check_package(root, record)
    sdk_record, _ = sdk_registered(options)
    if (
        sdk_record != record["sdk_registration"]
        or manifest["sdk_registration"] != sdk_record
    ):
        raise ValueError("ORT SDK/runtime identity mismatch; requalification required")
    return record, root


def main():
    args = sys.argv[1:]
    if args == ["--launcher-help"]:
        print(
            "Usage: bc250-ort [bc250-torch launcher options] SCRIPT.py [args] | --ort-info\n"
            "Explicit gfx1013 FP32 ONNX inference; supports daily/rccl-single, mounts and env.\n"
            "No --shell; no automatic Torch dispatch or ONNX autograd."
        )
        return
    daily = Path("/opt/bc250-stack/current")
    runtime = json.loads((daily / "runtime.json").read_text())
    if os.geteuid() != 0:
        if os.geteuid() != runtime["uid"]:
            raise ValueError("This runtime belongs to another host user")
        raise SystemExit(
            exit_status(run_process(["doas", str(Path(__file__).resolve()), *args]))
        )
    info_only = args == ["--ort-info"]
    options = parse_launcher_arguments([] if info_only else args, dict(os.environ))
    if options.shell or (not info_only and not options.arguments):
        raise ValueError("Supply Python arguments; --shell is not supported")
    with runtime_lock(daily / ".runtime.lock"):
        record, root = registered(options)
        if info_only:
            print(json.dumps(record, indent=2))
            return
        python_args, mounts = prepare_python_arguments(options.arguments, Path.cwd())
        command = [
            "/usr/local/bin/bc250-migraphx",
            "--runtime-profile",
            options.runtime_profile,
            "--shm-size",
            options.shm_size,
        ]
        for path, _ in mounts:
            command += ["--mount", str(path)]
        for path, mode in options.mounts:
            command += ["--mount-ro" if mode == "ro" else "--mount", path]
        for name, value in options.environment.items():
            command += ["--env", name + "=" + value]
        command += [
            "--mount-ro",
            str(root),
            str(root / "bin/bc250-ort.py"),
            *python_args,
        ]
        raise SystemExit(exit_status(run_process(command)))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        print("bc250-ort: " + str(error), file=sys.stderr)
        raise SystemExit(2)
