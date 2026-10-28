#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Record immutable source/dependency evidence and a bounded isolated ORT candidate plan."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile

REVISION = "402979ac1bef89dcd55add41eaa7277c407599eb"
SELECTED = (
    "abseil_cpp",
    "re2",
    "protobuf",
    "eigen",
    "date",
    "mp11",
    "json",
    "microsoft_gsl",
    "safeint",
    "flatbuffers",
    "onnx",
    "pybind11",
    "pytorch_cpuinfo",
)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    out = args.output.resolve()
    out.mkdir()

    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(source), *arguments])

    assert git("rev-parse", "HEAD").decode().strip() == REVISION
    assert not git("status", "--porcelain")
    checks = []

    def check(name, condition):
        assert condition, name
        checks.append(name)

    read = lambda name: (source / name).read_text()
    cmake = read("cmake/onnxruntime_providers_migraphx.cmake")
    ep = read("onnxruntime/core/providers/migraphx/migraphx_execution_provider.cc")
    info = read(
        "onnxruntime/core/providers/migraphx/migraphx_execution_provider_info.h"
    )
    lang = read("cmake/onnxruntime_language_standard_versions.cmake")
    check("pinned-version", read("VERSION_NUMBER").strip() == "1.31.0")
    check(
        "cmake-3-28-minimum",
        "cmake_minimum_required(VERSION 3.28)" in read("cmake/CMakeLists.txt"),
    )
    check(
        "cxx20-minimum",
        re.search(r"onnxruntime_MINIMUM_CXX_STANDARD_VERSION\s+20", lang) is not None,
    )
    check(
        "shared-provider-migraphx-c-and-hip-host",
        all(
            x in cmake
            for x in (
                "onnxruntime_add_shared_library(onnxruntime_providers_migraphx",
                "PRIVATE migraphx::c hip::host",
            )
        ),
    )
    check(
        "external-data-api-detection",
        "check_symbol_exists(migraphx_onnx_options_set_external_data_path" in cmake,
    )
    check("onnx-buffer-parser", "migraphx::parse_onnx_buffer" in ep)
    check(
        "native-stream-boundary",
        "prog.run_async(m, static_cast<hipStream_t>(rocm_stream))" in ep,
    )
    check("fallback-output-device-copy", "hipMemcpyWithStream(output_data" in ep)
    check(
        "fp32-defaults",
        all(
            re.search(r"bool\s+" + flag + r"\{false\}", info)
            for flag in (
                "fp16_enable",
                "bf16_enable",
                "fp8_enable",
                "int8_enable",
                "exhaustive_tune",
            )
        ),
    )
    check("fast-math-disabled", "co.set_fast_math(false)" in ep)
    check(
        "provider-only-export",
        "GetProvider;" in read("onnxruntime/core/providers/migraphx/version_script.lds")
        and "local:" in read("onnxruntime/core/providers/migraphx/version_script.lds"),
    )
    check(
        "cpu-fallback-disable-control",
        "disable_cpu_ep_fallback"
        in read("onnxruntime/core/session/inference_session.cc"),
    )
    dependencies = {}
    for line in read("cmake/deps.txt").splitlines():
        if line and not line.startswith("#"):
            name, url, digest = line.split(";")
            dependencies[name] = dict(url=url, upstream_sha1=digest)
    assert set(SELECTED).issubset(dependencies)
    selected = {n: dependencies[n] for n in SELECTED}
    (out / "dependencies.json").write_text(
        json.dumps(
            dict(
                status="source-pinned-not-fetched",
                selected_for_candidate=selected,
                complete_upstream_inventory=dependencies,
                note="Candidate list inferred from chosen native Linux flags; actual offline CMake configure remains required. Preserve upstream patches; add SHA256 when fetching bytes.",
            ),
            indent=2,
        )
        + "\n"
    )
    files, links, gitlinks = {}, {}, {}
    for entry in git("ls-files", "--stage", "-z").split(b"\0"):
        if not entry:
            continue
        metadata, name = entry.split(b"\t", 1)
        mode, blob, _ = metadata.decode().split()
        name = name.decode()
        path = source / name
        if mode == "160000":
            gitlinks[name] = blob
        elif path.is_symlink():
            links[name] = str(path.readlink())
        else:
            files[name] = sha(path)
    archive = out / "onnxruntime-source.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name in sorted(set(files) | set(links)):
            tar.add(source / name, arcname=name, recursive=False)
    symbols = sorted(
        set(
            re.findall(
                r"\bhip[A-Z][A-Za-z0-9_]*\s*(?=\()",
                "\n".join(
                    p.read_text()
                    for p in (source / "onnxruntime/core/providers/migraphx").glob(
                        "*.cc"
                    )
                ),
            )
        )
    )
    options = dict(
        onnxruntime_USE_MIGRAPHX="ON",
        AMD_MIGRAPHX_HOME="/sdk",
        onnxruntime_ENABLE_PYTHON="ON",
        onnxruntime_BUILD_SHARED_LIB="ON",
        onnxruntime_BUILD_UNIT_TESTS="OFF",
        onnxruntime_BUILD_BENCHMARKS="OFF",
        onnxruntime_ENABLE_TRAINING="OFF",
        onnxruntime_ENABLE_TRAINING_OPS="OFF",
        onnxruntime_USE_CUDA="OFF",
        onnxruntime_USE_TENSORRT="OFF",
        onnxruntime_USE_DNNL="OFF",
        onnxruntime_USE_XNNPACK="OFF",
        onnxruntime_USE_MIMALLOC="OFF",
        onnxruntime_USE_VCPKG="OFF",
        onnxruntime_USE_1DS_TELEMETRY="OFF",
        onnxruntime_USE_OPENMP="OFF",
        CMAKE_CXX_SCAN_FOR_MODULES="OFF",
        CMAKE_BUILD_TYPE="Release",
        CMAKE_CXX_COMPILER_LAUNCHER="ccache",
        CMAKE_C_COMPILER_LAUNCHER="ccache",
        CMAKE_JOB_POOLS="link_pool=1",
        CMAKE_JOB_POOL_LINK="link_pool",
    )
    plan = dict(
        status="planned-not-configured-or-built",
        cmake_options=options,
        python="Qualified image CPython 3.12 isolated build venv; preserve existing packages",
        source_read_only=True,
        fetch_network="Local fetch only; hash-verified offline mirror on .104",
        cpu_jobs=4,
        ram_bytes=10 * 1024**3,
        memory_plus_swap_bytes=12 * 1024**3,
        private_ccache_bytes=2 * 1024**3,
        phase_deadline_seconds=7200,
        disk_admission_ordinary_bytes=24 * 1024**3,
        growth_cap_bytes=20 * 1024**3,
        rescue_floor_ordinary_bytes=4 * 1024**3,
        footprint="Provisional conservative budget, not measured full ORT build footprint",
        candidate="Separate wheel/module prefix and explicit activation; do not overwrite CPU ORT, SDK registry, runtime image/manifests",
        dtype="FP32 only; explicitly disable low-precision provider options/env and exhaustive tuning",
        acceptance=[
            "Actual ORT session provider and profile node assignment, no silent CPU fallback",
            "Independent CPU-double/retained CPU ORT references, changed inputs, shape/type refusals",
            "Torch/CPU ORT/MIGraphX import orders and native stream/copy/allocator lifecycle",
            "Native SDK/runtime hashes, AOT caches, retained daily/RCCL and kernel-log guards",
        ],
        risks=[
            "C++20 wrapper source repair144f6db5c not in installed SDK; build uses verified private overlay",
            "ORT protobuf33.6/Abseil20250814 versus private MIGraphX protobuf30/Abseil20250512; inspect actual ELF visibility/binding and import-order tests",
            "Do not attempt to load an arbitrary provider DSO into a different ORT binary; provider bridge is internal ABI",
            "Source has quantization/dynamic-shape/cache paths beyond current admission; leave unqualified",
        ],
    )
    (out / "candidate-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    report = dict(
        status="source-audit-passed",
        revision=REVISION,
        version="1.31.0",
        checks=checks,
        source_files_sha256=files,
        source_symlinks=links,
        uninitialized_gitlinks=gitlinks,
        shallow_repository=git("rev-parse", "--is-shallow-repository").decode().strip()
        == "true",
        source_archive_sha256=sha(archive),
        source_archive_bytes=archive.stat().st_size,
        hip_call_names=symbols,
        upstream="https://github.com/microsoft/onnxruntime.git",
        tag_signature_qualified=False,
        provider_built=False,
        provider_installed=False,
    )
    (out / "source-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print("ORT_SOURCE_AUDIT_PASS", len(checks), len(files), len(links), flush=True)


if __name__ == "__main__":
    main()
