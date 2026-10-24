#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Record pinned MIGraphX source evidence without modifying upstream sources."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--rocm-cmake", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def git(root, *items):
        return subprocess.check_output(
            ["git", "-C", str(root), *items], text=True
        ).strip()

    assert (
        git(args.source, "rev-parse", "HEAD")
        == "95672916ed289d4be9d230a9be732e1bbad97e8c"
    )
    assert (
        git(args.rocm_cmake, "rev-parse", "HEAD")
        == "6a7c5b73b8882c74f8f7060e2633f230dabb7b63"
    )
    assert not git(args.source, "status", "--porcelain") and not git(
        args.rocm_cmake, "status", "--porcelain"
    )
    files = {
        n: hashlib.sha256((args.source / n).read_bytes()).hexdigest()
        for n in git(args.source, "ls-files").splitlines()
        if (args.source / n).is_file()
    }
    root = "src/targets/gpu/"
    device = (args.source / root / "device_name.cpp").read_text()
    desc = (args.source / root / "device_description.cpp").read_text()
    assert 'starts_with(gfx, "gfx10")' in desc and "return 32;" in desc
    assert 'return not(starts_with(gfx_name, "gfx1030"));' in device
    report = dict(
        status="source-audited",
        upstream="https://github.com/ROCm/AMDMIGraphX",
        tag="rocm-10.1",
        revision=git(args.source, "rev-parse", "HEAD"),
        tag_object_type=git(args.source, "cat-file", "-t", "rocm-10.1"),
        signature_qualification="Pinned commit/TLS source; lightweight tag, no local signature qualification",
        source_unmodified=True,
        source_files=len(files),
        source_sha256=files,
        rocm_cmake_revision=git(args.rocm_cmake, "rev-parse", "HEAD"),
        rocm_cmake_unmodified=True,
        findings={
            "wavefront": "Live HIP properties; cross-compile gfx10 inference is wave32",
            "generic_kernels": "Own pointwise/reduction headers have native HIP paths independent of CK/MLIR",
            "bf16": "Current predicate excludes gfx1030 only, reports true for gfx1013; admission needs dedicated graph/storage/backend validation, not a reproduced numerical failure",
            "fp8_mx": "gfx1013 excluded by existing FP8/MX architecture predicates",
            "hipblaslt": "MIGraphX whitelist excludes gfx1013; use rocBLAS first despite separate qualified scalar hipBLASLt component",
            "dpp_disabled": "C++17 warning from an unguarded subwave_reduce reference; strict JIT uses Werror; probe AOT fallback result alone does not qualify strict JIT",
            "initial_profile": "GPU/rocBLAS/MIOpen on; hipBLASLt/CK/MLIR/AMDMLSS/Eigen/Python/ONNX/TF off",
            "onnx_dependencies": "Python ONNX/protobuf wheels do not provide native C++ Protobuf/protoc/Abseil needed by the MIGraphX parser",
            "full_build": "Not attempted; compile/configure admission only. Need measured build-space budget before full graph backend or provider work",
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("MIGRAPHX_SOURCE_AUDIT_PASS", len(files))


if __name__ == "__main__":
    main()
