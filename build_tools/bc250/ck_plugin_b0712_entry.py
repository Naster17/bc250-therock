#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Activate the identity-pinned FP32/FP16 plugin and repaired MIOpen in a fresh worker."""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

NAME = "bc250-miopen-dl-gfx1013-b0712"
PREFIX = Path("/workspace/build/dist/rocm")
SONAMES = (
    "libMIOpen.so.1",
    "libamdhip64.so.7",
    "libhsa-runtime64.so.1",
    "libstdc++.so.6",
    "libm.so.6",
    "libgcc_s.so.1",
    "libc.so.6",
    "ld-linux-x86-64.so.2",
    "librocm_sysdeps_zstd.so.1",
    "librocm_sysdeps_bz2.so",
    "libhiprtc.so.7",
    "libamd_comgr.so.3",
    "librocblas.so.5",
    "libhipblaslt.so.1",
    "librocm_sysdeps_sqlite3.so",
    "libroctx64.so.4",
)


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def environment(root):
    return dict(
        os.environ,
        LD_LIBRARY_PATH=str(root / "runtime")
        + ":"
        + str(PREFIX / "lib")
        + ":"
        + str(PREFIX / "lib/rocm_sysdeps/lib")
        + ":"
        + os.environ.get("LD_LIBRARY_PATH", ""),
        MIOPEN_CK_LIB_PATH=str(root / "lib"),
    )


def runtime_identity():
    handles = [C.CDLL(name) for name in SONAMES]
    paths = {
        Path(line.split(None, 5)[5].strip()).resolve(strict=True)
        for line in Path("/proc/self/maps").read_text().splitlines()
        if len(line.split(None, 5)) == 6 and line.split(None, 5)[5].startswith("/")
    }
    result = {}
    for name in SONAMES:
        found = {p for p in paths if p.name == name or p.name.startswith(name + ".")}
        if len(found) != 1:
            raise ValueError("ambiguous/missing runtime library: " + name)
        path = found.pop()
        result[name] = dict(sha256=sha(path), path=str(path))
    # Pin additional mapped ROCm dependencies, not just the top-level SONAMEs.
    known = {v["path"] for v in result.values()}
    for path in sorted(paths):
        if (
            path.is_relative_to(PREFIX)
            and str(path) not in known
            and ".so" in path.name
        ):
            result[path.name] = dict(sha256=sha(path), path=str(path))
    assert handles
    return result


def verify_files(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest["format"] != 2
        or manifest["name"] != NAME
        or manifest["architecture"] != "gfx1013:xnack-"
    ):
        raise ValueError("unsupported artifact identity/architecture")
    for name, digest in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("escaping payload path")
        path = root / relative
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError("escaping/symlink payload")
        if sha(path) != digest:
            raise ValueError("payload hash mismatch: " + name)
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if actual != set(manifest["files"]) | {"manifest.json"} or any(
        p.is_symlink() for p in root.rglob("*")
    ):
        raise ValueError("unexpected/missing artifact payload")
    return manifest


def verify_runtime(root, manifest):
    native = runtime_identity()
    expected = manifest["runtime_libraries"]
    if set(native) != set(expected) or any(
        native[k]["sha256"] != expected[k]["sha256"] for k in native
    ):
        raise ValueError("native MIOpen/HIP/C++ runtime identity mismatch")
    if native["libMIOpen.so.1"]["path"] != str(root / "runtime/libMIOpen.so.1"):
        raise ValueError("MIOpen is not mapped from this artifact")
    import torch

    if torch.cuda.get_device_properties(0).gcnArchName != "gfx1013:xnack-":
        raise ValueError("requires native gfx1013:xnack-")
    miopen_paths = {
        line.split(None, 5)[5].strip()
        for line in Path("/proc/self/maps").read_text().splitlines()
        if len(line.split(None, 5)) == 6 and "/libMIOpen.so." in line
    }
    if miopen_paths != {str(root / "runtime/libMIOpen.so.1")}:
        raise ValueError("conflicting MIOpen mapping after Torch import")
    plugin = C.CDLL(str(root / "lib/libMIOpenCKGroupedConv_gfx1013.so"))
    if plugin.ck_impl_get_api_version() != 5 or not all(
        getattr(plugin, n, None) for n in manifest["required_symbols"]
    ):
        raise ValueError("plugin API5/export mismatch")
    return native


def main():
    worker = sys.argv[1:2] == ["--worker"]
    if worker:
        del sys.argv[1]
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("check")
    seed = sub.add_parser("seed-db")
    seed.add_argument("--cohort", choices=("daily", "rccl"), required=True)
    seed.add_argument("--destination", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("script", type=Path)
    run.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        root = Path(__file__).resolve().parent.parent
        manifest = verify_files(root)
        if not worker:
            # The outer launcher may already own a HIP/KFD context. Keep that
            # parent alive and start a separate worker; never exec-replace it.
            code = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    *sys.argv[1:],
                ],
                env=environment(root),
            ).returncode
            raise SystemExit(code)
        native = verify_runtime(root, manifest)
        if args.action == "check":
            print(
                json.dumps(
                    dict(
                        status="passed",
                        artifact=NAME,
                        architecture=manifest["architecture"],
                        api=5,
                        runtime_libraries=native,
                    )
                )
            )
        elif args.action == "seed-db":
            shutil.copytree(root / "examples" / args.cohort, args.destination)
        else:
            script = args.script.resolve(strict=True)
            os.environ["MIOPEN_CK_LIB_PATH"] = str(root / "lib")
            sys.argv = [str(script), *args.arguments]
            sys.path[0] = str(script.parent)
            runpy.run_path(str(script), run_name="__main__")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        print("bc250-ck: " + str(error), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
