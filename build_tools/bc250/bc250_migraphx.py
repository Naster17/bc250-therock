#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Activate the registered gfx1013 inference SDK through bc250-torch, explicitly."""
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

from bc250_torch import (
    exit_status,
    parse_launcher_arguments,
    prepare_python_arguments,
    run_process,
    runtime_lock,
    selected_release,
)

BASE = Path("/opt/bc250-stack/sdks")
REGISTRY = BASE / "migraphx.json"
NAME = "bc250-migraphx-2.18.0.dev-gfx1013-cp312-b067"
INDEX_SHA = "9b0311b2a57790e42a76eac35a615b331efd224e81f5c7d1c2aa9dccb3d7ca5c"
MANIFEST_SHA = "75acf26290977370bfd739753ea860ebcb817716e869888edadbf08c7b14e25a"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def trusted(path):
    """Only root-controlled files/directories may control elevated activation."""
    for item in (path, *path.parents):
        info = item.lstat()
        if info.st_uid != 0 or info.st_mode & 0o022 or stat.S_ISLNK(info.st_mode):
            raise ValueError(f"Untrusted SDK path: {item}")


def check_sdk(root):
    trusted(root)
    info = root / "share/bc250-sdk"
    if (
        sha(info / "sha256.json") != INDEX_SHA
        or sha(info / "manifest.json") != MANIFEST_SHA
    ):
        raise ValueError("SDK index/manifest identity mismatch")
    index = json.loads((info / "sha256.json").read_text())
    expected = (
        set(index["files"]) | set(index["symlinks"]) | {"share/bc250-sdk/sha256.json"}
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
                raise ValueError(f"SDK link identity mismatch: {name}")
            if path.lstat().st_uid != 0:
                raise ValueError(f"Untrusted SDK link: {name}")
            actual.add(name)
        else:
            trusted(path)
            if path.is_file():
                actual.add(name)
                if name in index["files"] and sha(path) != index["files"][name]:
                    raise ValueError(f"SDK file identity mismatch: {name}")
            elif not path.is_dir():
                raise ValueError(f"Unsupported SDK entry: {name}")
    if actual != expected:
        raise ValueError("SDK file set mismatch")
    return json.loads((info / "manifest.json").read_text())


def registered(options):
    trusted(REGISTRY)
    record = json.loads(REGISTRY.read_text())
    root = Path(record["sdk"])
    if root != BASE / NAME or record["index_sha256"] != INDEX_SHA:
        raise ValueError("Unexpected SDK registration")
    check_sdk(root)
    release = selected_release(options)
    runtime = release / "runtime.json"
    if sha(runtime) != record["runtime_sha256"][options.runtime_profile]:
        raise ValueError("SDK runtime identity mismatch; requalification required")
    if json.loads(runtime.read_text())["image"] != record["runtime_image"]:
        raise ValueError("SDK runtime image mismatch")
    return record, root


def main():
    arguments = sys.argv[1:]
    if arguments == ["--launcher-help"]:
        print(
            "Usage: bc250-migraphx [bc250-torch launcher options] SCRIPT.py [args]\n"
            "       bc250-migraphx --sdk-info\n"
            "Explicit FP32 inference SDK; daily (default) or --runtime-profile rccl-single.\n"
            "Supports --mount, --mount-ro, --env, --env-file, --shm-size. No --shell.\n"
            "Uses the qualified CPython 3.12 image; no automatic Torch or ORT routing."
        )
        return
    daily = Path("/opt/bc250-stack/current")
    manifest = json.loads((daily / "runtime.json").read_text())
    if os.geteuid() != 0:
        if os.geteuid() != manifest["uid"]:
            raise ValueError("This runtime belongs to another host user")
        raise SystemExit(
            exit_status(
                run_process(["doas", str(Path(__file__).resolve()), *arguments])
            )
        )
    info_only = arguments == ["--sdk-info"]
    options = parse_launcher_arguments([] if info_only else arguments, dict(os.environ))
    if options.shell or (not info_only and not options.arguments):
        raise ValueError("Supply Python arguments; --shell is not supported")
    with runtime_lock(daily / ".runtime.lock"):
        record, root = registered(options)
        if info_only:
            print(json.dumps(record, indent=2))
            return
        # The SDK wrapper becomes Python's first script. Preserve the user's actual
        # script spelling and parent mounts before delegating to the ordinary launcher.
        python_args, mounts = prepare_python_arguments(options.arguments, Path.cwd())
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
            str(root / "bin/bc250-migraphx.py"),
            *python_args,
        ]
        raise SystemExit(exit_status(run_process(command)))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        print(f"bc250-migraphx: {error}", file=sys.stderr)
        raise SystemExit(2)
