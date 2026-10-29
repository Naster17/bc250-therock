#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Explicit in-runtime activation for the exact-ABI BC-250 DL artifact."""
import argparse
import ctypes as C
import hashlib
import json
import os
import runpy
from pathlib import Path
import shutil
import sys

SONAMES = (
    "libMIOpen.so.1",
    "libamdhip64.so.7",
    "libhsa-runtime64.so.1",
    "libstdc++.so.6",
    "libm.so.6",
    "libgcc_s.so.1",
    "libc.so.6",
    "ld-linux-x86-64.so.2",
)


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def runtime_identity():
    # Keep handles live while resolving the actual mapped objects, before Torch import.
    handles = [C.CDLL(name) for name in SONAMES]
    paths = {
        Path(line.split(None, 5)[5].strip())
        for line in Path("/proc/self/maps").read_text().splitlines()
        if len(line.split(None, 5)) == 6 and line.split(None, 5)[5].startswith("/")
    }
    result = {}
    for name in SONAMES:
        found = {
            p.resolve(strict=True)
            for p in paths
            if p.name == name or p.name.startswith(name + ".")
        }
        if len(found) != 1:
            raise ValueError("ambiguous/missing runtime library: " + name)
        path = found.pop()
        result[name] = dict(sha256=sha(path), path=str(path))
    assert handles
    return result


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest["format"] != 1
        or manifest["name"] != "bc250-miopen-dl-gfx1013-b077"
        or manifest["architecture"] != "gfx1013:xnack-"
    ):
        raise ValueError("unsupported artifact identity/architecture")
    for name, digest in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("escaping payload path")
        path = root / relative
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(
            root.resolve()
        ):
            raise ValueError("escaping/symlink payload")
        if sha(path) != digest:
            raise ValueError("payload hash mismatch: " + name)
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if actual != set(manifest["files"]) | {"manifest.json"} or any(
        p.is_symlink() for p in root.rglob("*")
    ):
        raise ValueError("unexpected/missing artifact payload")
    native = runtime_identity()
    if set(manifest["runtime_libraries"]) != set(SONAMES) or any(
        native[k]["sha256"] != manifest["runtime_libraries"][k]["sha256"]
        for k in SONAMES
    ):
        raise ValueError("native MIOpen/HIP/C++ runtime identity mismatch")
    import torch

    if torch.cuda.get_device_properties(0).gcnArchName != "gfx1013:xnack-":
        raise ValueError("requires native gfx1013:xnack-")
    plugin = C.CDLL(str(root / "lib/libMIOpenCKGroupedConv_gfx1013.so"))
    if plugin.ck_impl_get_api_version() != 5 or not all(
        getattr(plugin, n, None) for n in manifest["required_symbols"]
    ):
        raise ValueError("plugin API5/export mismatch")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("check")
    capture = sub.add_parser("capture-runtime")
    capture.add_argument("--output", type=Path, required=True)
    seed = sub.add_parser("seed-db")
    seed.add_argument("--cohort", choices=("daily", "rccl"), required=True)
    seed.add_argument("--destination", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("script", type=Path)
    run.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        if args.action == "capture-runtime":
            args.output.write_text(json.dumps(runtime_identity(), indent=2) + "\n")
            return
        root = Path(__file__).resolve().parent.parent
        manifest = verify(root)
        if args.action == "check":
            print(
                json.dumps(
                    dict(
                        status="passed",
                        artifact=manifest["name"],
                        architecture=manifest["architecture"],
                        api=5,
                    )
                )
            )
        elif args.action == "seed-db":
            # Never merge into an existing application's tuning cache.
            shutil.copytree(root / "examples" / args.cohort, args.destination)
        else:
            script = args.script.resolve(strict=True)
            os.environ["MIOPEN_CK_LIB_PATH"] = str(root / "lib")
            # HIP was initialized by native architecture validation. Preserve that
            # process/mm rather than exec-replacing a live KFD context.
            sys.argv = [str(script), *args.arguments]
            sys.path[0] = str(script.parent)
            runpy.run_path(str(script), run_name="__main__")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        print("bc250-ck: " + str(error), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
