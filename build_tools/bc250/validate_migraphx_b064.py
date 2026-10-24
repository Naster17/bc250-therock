#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run isolated FP32 MIGraphX graphs, retaining explicit native CPU references."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    ws = Path(__file__).resolve().parent.parent
    storage = ws / "tmp/migraphx-b064/20261025-storage/mount"
    build = storage / "build-r1"
    scratch = storage / "runtime-r2"
    assert scratch.is_dir() and os.access(scratch, os.W_OK)
    env = dict(os.environ)
    env.update(
        TMPDIR=str(scratch),
        TMP=str(scratch),
        TEMP=str(scratch),
        MIOPEN_USER_DB_PATH=str(scratch / "miopen-db"),
        MIOPEN_CUSTOM_CACHE_DIR=str(scratch / "miopen-cache"),
    )
    env["LD_LIBRARY_PATH"] = (
        str(build / "migraphx/lib")
        + ":"
        + env.get("LD_LIBRARY_PATH", "")
        + ":/workspace/build/dist/rocm/lib/rocm_sysdeps/lib"
    )
    report = dict(
        status="running", checks=[], inference_only=True, scratch=str(scratch)
    )
    try:
        loader = subprocess.run(
            [
                "/lib64/ld-linux-x86-64.so.2",
                "--list",
                str(build / "probe_migraphx_b064"),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        (args.output.parent / "loader.log").write_text(loader.stdout + loader.stderr)
        assert (
            loader.returncode == 0 and "not found" not in loader.stdout
        ), loader.stderr
        assert str(build / "migraphx/lib") in loader.stdout
        assert (
            "/workspace/build/dist/rocm/lib/rocm_sysdeps/lib/librocm_sysdeps_sqlite3.so"
            in loader.stdout
        )
        completed = subprocess.run(
            [str(build / "probe_migraphx_b064")],
            capture_output=True,
            text=True,
            timeout=480,
            env=env,
        )
        (args.output.parent / "graph-stdout.log").write_text(completed.stdout)
        (args.output.parent / "graph-stderr.log").write_text(completed.stderr)
        assert completed.returncode == 0, completed.stderr
        result = json.loads(completed.stdout.splitlines()[-1])
        assert result["status"] == "passed" and result["checks"] == 22
        report["checks"] = [
            "native-gfx1013-wave32",
            "rocblas-only-provider-query-overloads",
        ]
        for group, sizes in [
            ("pointwise", [17, 33]),
            ("reduce", [17, 33, 129, 513]),
            ("rocblas-gemm", [7, 17]),
            ("miopen-conv", [1, 2]),
        ]:
            report["checks"] += [
                f"{group}-{n}-cpu-double-guards-replay{r}"
                for n in sizes
                for r in (0, 1)
            ]
        assert len(report["checks"]) == 22
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
