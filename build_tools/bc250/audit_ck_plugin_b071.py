#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Inventory pinned MIOpen plugin admission and existing scalar CK factory sources."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--libraries", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = args.libraries
    paths = dict(
        plugin_cmake="projects/miopen/src/ck_impl/CMakeLists.txt",
        loader="projects/miopen/src/solver/ck_impl_lib_loader.cpp",
        interface="projects/miopen/src/include/miopen/solver/ck_impl_interface.hpp",
        whitelist="projects/miopen/src/include/miopen/solver/ck_utility_common.hpp",
        plugin_forward="projects/miopen/src/ck_impl/ck_grouped_conv_fwd_impl.cpp",
        compile_defs="projects/miopen/cmake/MiopenCkCompileDefs.cmake",
        factory="projects/composablekernel/library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp",
        dl_instance="projects/composablekernel/library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd/nhwgc/dl/device_grouped_conv2d_fwd_dl_nhwgc_gkyxc_nhwgk_f32_instance.cpp",
        ck_cmake="projects/composablekernel/CMakeLists.txt",
    )
    text = {k: (root / v).read_text() for k, v in paths.items()}
    version = int(re.search(r"#define CK_IMPL_API_VERSION (\d+)", text["interface"])[1])
    archs = re.search(
        r"set\(_CK_SUPPORTED_ARCHS\s+(.*?)\)", text["plugin_cmake"], re.S
    )[1].split()
    prefixes = re.findall(r'\{CKSolverType::\w+, "([a-z0-9_]+)"\}', text["loader"])
    common = re.findall(r'bind_symbol\(\w+, "(ck_impl_[a-z_]+)"\)', text["loader"])
    suffixes = re.findall(r'\(sym \+ "([a-z_]+)"\)', text["loader"])
    required = sorted(
        set(common + [f"ck_impl_{a}_{b}" for a in prefixes for b in suffixes])
    )
    assert version == 5 and len(prefixes) == 12 and len(required) == 66
    assert "gfx1013" not in archs
    assert "gfx1013" not in text["whitelist"]
    assert "env_path.empty()" in text["loader"] and "is_ck_whitelist" in text["loader"]
    assert "DeviceGroupedConvFwdMultipleABD" in text["plugin_forward"]
    assert "convolution::NHWGC" in text["plugin_forward"]
    assert "#ifdef DL_KERNELS" in text["factory"]
    assert (
        "add_device_grouped_conv2d_fwd_dl_nhwgc_gkyxc_nhwgk_f32_instances"
        in text["factory"]
    )
    assert "DeviceGroupedConvFwdMultipleABD" in text["dl_instance"]
    assert "add_definitions(-DDL_KERNELS)" in text["ck_cmake"]
    assert "DL_KERNELS" not in text["compile_defs"]
    report = dict(
        status="passed-source-admission-audit",
        api_version=version,
        plugin_supported_archs=archs,
        required_solver_groups=prefixes,
        required_symbols=required,
        required_symbol_count=len(required),
        abi="extern-C entry names carry internal C++ ProblemDescription/ExecutionContext/ConvSolution types; matching MIOpen build required",
        default_gfx1013_admitted=False,
        private_plugin_path_bypasses_default_skip=True,
        scalar_forward_factory="Existing DL FP32/FP16 2D NHWGC/GKYXC/NHWGK factory branch matches plugin forward types",
        macro_boundary="Factory requires DL_KERNELS; CK_ENABLE_DL_KERNELS alone does not select that branch. MIOpen compile-def helper does not explicitly propagate DL_KERNELS; actual private plugin compile flags must be checked.",
        source_sha256={
            v: hashlib.sha256((root / v).read_bytes()).hexdigest()
            for v in paths.values()
        },
        decision="B07.2 isolated matching-build forward-only FP32 DL plugin prototype; export all required ABI groups, explicitly refuse unsupported directions/types, no default whitelist or production installation change",
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                status=report["status"],
                api_version=version,
                required_symbols=len(required),
                sources=len(paths),
            )
        )
    )


if __name__ == "__main__":
    main()
