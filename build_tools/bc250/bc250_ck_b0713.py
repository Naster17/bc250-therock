#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Activate the registered B07.12 coherent FP32/FP16 MIOpen DL artifact explicitly.

B07.13 variant of bc250_ck.py: identical format-agnostic caller-context/lock/
delegation logic, pinned to the B07.12 coherent artifact (format2 worker entry,
private repaired MIOpen, 27 mapped library identities). Preserves ordinary
defaults; the old B07.7 FP32 registration is refused by this wrapper and retained
as the live-rollback target.
"""
import json
import os
from pathlib import Path
import sys
import tempfile

from bc250_migraphx import sha, trusted
from bc250_torch import (
    exit_status,
    parse_launcher_arguments,
    prepare_python_arguments,
    run_process,
    runtime_lock,
    read_context,
    selected_release,
)

BASE = Path("/opt/bc250-stack/sdks")
NAME = "bc250-miopen-dl-gfx1013-b0712"
REGISTRY = BASE / "miopen-dl.json"
MANIFEST_SHA = "52db5d4262bd29e6307b13a5862cc0652f0e0b6f9c8509761d03d39965e02846"
ARCHIVE_SHA = "a0d47ed4ba4df354cc7ea898a285eefa051f49693667386567ea392f66ddd91e"


def check_package(root, record):
    trusted(root)
    if (
        sha(root / "manifest.json") != MANIFEST_SHA
        or record["manifest_sha256"] != MANIFEST_SHA
    ):
        raise ValueError("CK manifest identity mismatch")
    manifest = json.loads((root / "manifest.json").read_text())
    expected = set(manifest["files"]) | {"manifest.json"}
    actual = set()
    for path in root.rglob("*"):
        trusted(path)
        if path.is_file():
            name = str(path.relative_to(root))
            actual.add(name)
            if name in manifest["files"] and sha(path) != manifest["files"][name]:
                raise ValueError("CK payload identity mismatch: " + name)
        elif not path.is_dir():
            raise ValueError("Unsupported CK package entry")
    if actual != expected:
        raise ValueError("CK package file set mismatch")
    return manifest


def registered(options):
    trusted(REGISTRY)
    record = json.loads(REGISTRY.read_text())
    root = Path(record["package"])
    if root != BASE / NAME:
        raise ValueError("Unexpected CK registration")
    if record["archive_sha256"] != ARCHIVE_SHA:
        raise ValueError("Unexpected CK archive identity")
    check_package(root, record)
    runtime = selected_release(options) / "runtime.json"
    if sha(runtime) != record["runtime_sha256"][options.runtime_profile]:
        raise ValueError("CK runtime identity mismatch; requalification required")
    if (
        json.loads(runtime.read_text())["image"]
        != record["runtime_images"][options.runtime_profile]
    ):
        raise ValueError("CK runtime image mismatch")
    return record, root


def main():
    args = sys.argv[1:]
    if args == ["--launcher-help"]:
        print(
            "Usage: bc250-ck [bc250-torch options] SCRIPT.py [args] | --check | --seed-db NEWDIR\n"
            "       bc250-ck --ck-info\n"
            "Explicit exact-ABI FP32/FP16 2D MIOpen DL forward; daily/rccl-single.\n"
            "Supports mounts/env/env-file/shm-size. No --shell, -c or -m.\n"
            "No global tuning/default change. seed-db never merges existing data."
        )
        return
    daily = Path("/opt/bc250-stack/current")
    runtime = json.loads((daily / "runtime.json").read_text())
    if os.geteuid() != 0:
        if os.geteuid() != runtime["uid"]:
            raise ValueError("This runtime belongs to another host user")
        if args == ["--ck-info"]:
            raise SystemExit(
                exit_status(run_process(["doas", str(Path(__file__).resolve()), *args]))
            )
        options = parse_launcher_arguments(args, dict(os.environ))
        if options.shell or not options.arguments:
            raise ValueError("Supply Python arguments; --shell is not supported")
        with tempfile.TemporaryDirectory(prefix="bc250-ck-context-") as directory:
            path = Path(directory) / "context.json"
            path.touch(mode=0o600)
            path.write_text(
                json.dumps(
                    {"options": vars(options), "workspace": str(Path.cwd().resolve())}
                )
            )
            status = run_process(
                ["doas", str(Path(__file__).resolve()), "--_context", str(path)]
            )
        raise SystemExit(exit_status(status))
    if args[:1] == ["--_context"]:
        if len(args) != 2:
            raise ValueError("Invalid private launcher context")
        options, workspace = read_context(Path(args[1]), runtime["uid"])
        info_only = False
    else:
        info_only = args == ["--ck-info"]
        options = parse_launcher_arguments([] if info_only else args, dict(os.environ))
        workspace = Path.cwd().resolve()
    if options.shell or (not info_only and not options.arguments):
        raise ValueError("Supply Python arguments; --shell is not supported")
    with runtime_lock(daily / ".runtime.lock"):
        record, root = registered(options)
        if info_only:
            print(json.dumps(record, indent=2))
            return
        if options.arguments == ["--check"]:
            python_args = ["check"]
            mounts = []
        elif len(options.arguments) == 2 and options.arguments[0] == "--seed-db":
            target = Path(options.arguments[1]).expanduser()
            if not target.is_absolute():
                target = workspace / target
            target = target.resolve()
            if target.exists():
                raise ValueError("Seed destination already exists")
            if not target.parent.is_dir():
                raise ValueError("Seed parent must exist")
            python_args = [
                "seed-db",
                "--cohort",
                "daily" if options.runtime_profile == "daily" else "rccl",
                "--destination",
                str(target),
            ]
            mounts = [(str(target.parent), str(target.parent))]
        else:
            if options.arguments[0].startswith("-"):
                raise ValueError("Supply a Python script, --check or --seed-db NEWDIR")
            script = Path(options.arguments[0]).expanduser()
            if not script.is_absolute():
                script = workspace / script
            if not script.is_file():
                raise ValueError("Python script must exist")
            python_args, mounts = prepare_python_arguments(
                [str(script.resolve()), *options.arguments[1:]], workspace
            )
            python_args = ["run", *python_args]
        command = [
            "/usr/local/bin/bc250-torch",
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
            str(root / "bin/bc250-ck.py"),
            *python_args,
        ]
        raise SystemExit(exit_status(run_process(command)))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        print("bc250-ck: " + str(error), file=sys.stderr)
        raise SystemExit(2)
