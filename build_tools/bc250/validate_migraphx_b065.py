#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Execute isolated native ONNX graphs and retained hand-built FP32 graphs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--group", choices=("onnx", "retained"), required=True)
    args = parser.parse_args()
    ws = Path(__file__).resolve().parent.parent
    storage = ws / "tmp/migraphx-b064/20261025-storage/mount/b065"
    build = storage / "build-r1"
    scratch = storage / ("runtime-r1-" + args.group)
    assert scratch.is_dir() and os.access(scratch, os.W_OK)
    generation = json.loads((storage / "models-generation-r2.json").read_text())
    assert generation["status"] == "passed" and len(generation["checks"]) == 12
    for name, digest in generation["sha256"].items():
        assert (
            hashlib.sha256((storage / "models-r2" / name).read_bytes()).hexdigest()
            == digest
        ), name
    env = dict(
        os.environ,
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
    executable = build / (
        "probe_migraphx_b065" if args.group == "onnx" else "probe_migraphx_b064"
    )
    report = dict(status="running", checks=[], group=args.group, inference_only=True)
    try:
        loader = subprocess.run(
            ["/lib64/ld-linux-x86-64.so.2", "--list", str(executable)],
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
        if args.group == "onnx":
            assert "libmigraphx_onnx.so" in loader.stdout
        command = [str(executable)] + (
            [str(storage / "models-r2")] if args.group == "onnx" else []
        )
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=480, env=env
        )
        (args.output.parent / "native-stdout.log").write_text(result.stdout)
        (args.output.parent / "native-stderr.log").write_text(result.stderr)
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout.splitlines()[-1])
        count = 24 if args.group == "onnx" else 22
        assert data["status"] == "passed" and data["checks"] == count
        if args.group == "onnx":
            report["checks"] = ["native-gfx1013-wave32", "rocblas-only-query"]
            for name in (
                "pointwise",
                "reduce",
                "dense",
                "transpose",
                "conv",
                "constant",
            ):
                report["checks"] += [
                    name + "-file-buffer-IR",
                    name + "-cpu-double-ort-replay0",
                    name + "-cpu-double-ort-replay1",
                ]
            report["checks"] += [
                f"{name}-{route}-refusal"
                for name in ("unknown", "corrupt")
                for route in ("file", "buffer")
            ]
        else:
            report["checks"] = [
                "retained-hand-built-FP32-" + str(i) for i in range(count)
            ]
        assert len(report["checks"]) == count
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
