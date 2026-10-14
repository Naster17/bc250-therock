#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build and execute focused scalar/generic GPU qualification in the developer tree."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import time
import xml.etree.ElementTree as et


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument(
        "--group",
        choices=(
            "scalar",
            "lt-priority1",
            "ck",
            "ck-priority1",
            "rpp",
            "tensor",
            "dnn",
        ),
        required=True,
    )
    args = parser.parse_args()
    ws = args.workspace.resolve()
    logs = ws / "logs/library-expansion"
    logs.mkdir(exist_ok=True)
    prefix = "/workspace/build/dist/rocm"
    source = "/workspace/source/rocm-libraries"
    report = {"group": args.group, "status": "running", "checks": []}
    output = logs / f"{args.group}.json"
    base = [
        "podman",
        "run",
        "--rm",
        "--network",
        "none",
        "--device",
        "/dev/kfd",
        "--device",
        "/dev/dri/renderD128",
        "-v",
        f"{ws / 'build'}:/workspace/build:rw",
        "-v",
        f"{ws / 'source'}:/workspace/source:ro",
        "-v",
        f"{ws / 'helpers'}:/workspace/helpers:ro",
        "-v",
        f"{logs}:/workspace/logs:rw",
        "-e",
        "HSA_ENABLE_SDMA=0",
        "-e",
        "OMP_NUM_THREADS=4",
        "-e",
        f"LD_LIBRARY_PATH={prefix}/lib:{prefix}/lib/llvm/lib",
        "-e",
        f"ROCM_PATH={prefix}",
        "-e",
        f"HIP_PATH={prefix}",
        "localhost/bc250-rocm-build:10.2-ccache",
    ]
    assert Path("/sys/class/watchdog/watchdog0/state").read_text().strip() == "active"

    def run(
        label: str, command: list[str], *, xml=False, marker=None, timeout=180
    ) -> None:
        start = time.monotonic()
        with (logs / f"{label}.log").open("w") as stream:
            # Bound the application inside the container, so a host timeout cannot leave GPU work running.
            result = subprocess.run(
                base + ["timeout", "--kill-after=10s", f"{timeout}s"] + command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=timeout + 30,
            )
        assert result.returncode == 0, f"{label}: see {logs / (label + '.log')}"
        count = 1
        if xml:
            root = et.parse(logs / f"{label}.xml").getroot()
            count = int(root.attrib["tests"])
            assert count > 0, root.attrib
            for key in ("failures", "errors", "disabled"):
                assert int(root.attrib.get(key, 0)) == 0, root.attrib
            for case in root.iter("testcase"):
                assert (
                    case.attrib.get("result") == "completed"
                    and case.find("skipped") is None
                ), case.attrib
        if marker:
            log = (logs / f"{label}.log").read_text()
            assert marker in log, label
            match = re.search(r"cases=(\d+)", log)
            if match:
                count = int(match.group(1))
        report["checks"].append(
            {
                "name": label,
                "executed_tests": count,
                "seconds": time.monotonic() - start,
            }
        )
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"PASS {label} ({count})", flush=True)

    def compile_probe(name: str, library: str) -> None:
        run(
            f"compile-{name}",
            [
                "bash",
                "-c",
                f"mkdir -p /workspace/build/bc250-coverage; "
                f"{prefix}/bin/hipcc -O2 -std=c++17 --offload-arch=gfx1013 "
                f"-I{prefix}/include -L{prefix}/lib /workspace/helpers/{name}.cpp "
                f"-l{library} -o /workspace/build/bc250-coverage/{name}",
            ],
            timeout=240,
        )

    subprocess.run(
        ["dmesg"],
        stdout=(logs / f"kernel-before-{args.group}.log").open("w"),
        check=True,
    )
    try:
        if args.group == "scalar":
            for name, library, marker in (
                ("hipblaslt_scalar_smoke", "hipblaslt", "HIPBLASLT_SCALAR_PASS"),
                ("hiptensor_scalar_smoke", "hiptensor", "HIPTENSOR_SCALAR_PASS"),
                ("hipsparselt_scalar_smoke", "hipsparselt", "HIPSPARSELT_SCALAR_PASS"),
            ):
                compile_probe(name, library)
                run(name, [f"/workspace/build/bc250-coverage/{name}"], marker=marker)
        elif args.group == "lt-priority1":
            for name, marker in (
                ("hipblaslt_scalar_smoke", "HIPBLASLT_SCALAR_PASS"),
                ("hipblaslt_priority1_smoke", "HIPBLASLT_PRIORITY1_PASS"),
            ):
                compile_probe(name, "hipblaslt")
                run(name, [f"/workspace/build/bc250-coverage/{name}"], marker=marker)
        elif args.group == "rpp":
            compile_probe("rpp_extended_smoke", "rpp")
            run(
                "rpp_extended_smoke",
                ["/workspace/build/bc250-coverage/rpp_extended_smoke"],
                marker="RPP_EXTENDED_NATIVE_GFX1013_PASS",
            )
        elif args.group == "ck-priority1":
            path = "/workspace/build/bc250-coverage/ck-tile-priority1"
            for name, marker in (
                ("topk", "CK_TILE_TOPK_PASS"),
                ("layernorm", "CK_TILE_LAYERNORM_PASS"),
                ("scalar_gemm", "CK_TILE_SCALAR_GEMM_PASS"),
            ):
                run(f"ck-p1-{name}", [f"{path}/{name}"], marker=marker)
            for dtype in ("fp16", "bf16", "fp32"):
                for operation in ("max", "min", "sum"):
                    run(
                        f"ck-p1-pool-{dtype}-{operation}",
                        [
                            f"{path}/pool3d",
                            f"-prec={dtype}",
                            f"-op={operation}",
                            "-N=1",
                            "-D=5",
                            "-H=7",
                            "-W=9",
                            "-C=17",
                            "-Z=2",
                            "-Y=3",
                            "-X=2",
                            "-Sz=1",
                            "-Sy=2",
                            "-Sx=2",
                            "-Dz=2",
                            "-Dy=1",
                            "-Dx=2",
                            "-LeftPz=1",
                            "-LeftPy=1",
                            "-LeftPx=1",
                            "-RightPz=0",
                            "-RightPy=1",
                            "-RightPx=0",
                            "-warmup=0",
                            "-repeat=1",
                            "-v=1",
                        ],
                        marker="valid:y",
                    )
        elif args.group == "ck":
            path = "/workspace/build/bc250-coverage/ck-tile"
            for dtype in ("fp16", "bf16", "fp32"):
                for n, stride in ((17, 24), (129, 136)):
                    run(
                        f"ck-rmsnorm-{dtype}-{n}",
                        [
                            f"{path}/rmsnorm",
                            "-m=7",
                            f"-n={n}",
                            f"-stride={stride}",
                            f"-prec={dtype}",
                            "-v=1",
                            "-warmup=0",
                            "-repeat=1",
                        ],
                        marker="valid:y",
                    )
            for dtype in ("fp16", "bf16"):
                run(
                    f"ck-reduce-{dtype}",
                    [
                        f"{path}/reduce",
                        "-n=7",
                        "-h=3",
                        "-w=5",
                        "-c=17",
                        f"-prec={dtype}",
                        "-v=1",
                        "-warmup=0",
                        "-repeat=1",
                    ],
                    marker="valid:y",
                )
            for dtype in ("fp16", "fp32"):
                for shape, perm in (("3,5,7", "2,0,1"), ("2,3,5,7", "0,3,1,2")):
                    run(
                        f"ck-permute-{dtype}-{shape}",
                        [
                            f"{path}/permute",
                            f"-shape={shape}",
                            f"-perm={perm}",
                            f"-prec={dtype}",
                            "-v=1",
                            "-warmup=0",
                            "-repeat=1",
                        ],
                    )
                run(
                    f"ck-transpose-{dtype}",
                    [
                        f"{path}/transpose",
                        "-m=17",
                        "-n=31",
                        "-stride_in=40",
                        f"-prec={dtype}",
                        "-v=1",
                        "-warmup=0",
                        "-repeat=1",
                    ],
                )
            for operation in ("square", "tanh", "fastgelu", "convert"):
                run(
                    f"ck-unary-{operation}",
                    [
                        f"{path}/unary",
                        "-m=17",
                        "-n=31",
                        "-stride=40",
                        f"-op={operation}",
                        "-x_prec=fp16",
                        "-y_prec=fp16",
                        "-v=1",
                        "-warmup=0",
                        "-repeat=1",
                    ],
                )
            run(
                "ck-pool3d",
                [
                    f"{path}/pool3d",
                    "-N=2",
                    "-D=5",
                    "-H=7",
                    "-W=9",
                    "-C=32",
                    "-v=1",
                    "-warmup=0",
                    "-repeat=1",
                ],
                marker="valid:y",
            )
        elif args.group == "tensor":
            for rank in (3, 4):
                for op, config in (
                    ("reduction", f"03_reduction/rank{rank}_test_params.yaml"),
                    (
                        "elementwise_permute",
                        f"02_elementwise/permutation/rank{rank}_test_params.yaml",
                    ),
                    (
                        "elementwise_binary_op",
                        f"02_elementwise/binary_op/rank{rank}_binary_op_test_params.yaml",
                    ),
                    (
                        "elementwise_trinary_op",
                        f"02_elementwise/trinary_op/rank{rank}_trinary_op_test_params.yaml",
                    ),
                ):
                    label = f"tensor-rank{rank}-{op}"
                    run(
                        label,
                        [
                            "env",
                            f"GTEST_OUTPUT=xml:/workspace/logs/{label}.xml",
                            f"{prefix}/bin/rank{rank}_{op}_test",
                            "-y",
                            f"{prefix}/bin/hiptensor/configs/validation/quick/{config}",
                            "-v",
                            "ON",
                            "--hot_runs",
                            "1",
                            "--cold_runs",
                            "0",
                        ],
                        xml=True,
                    )
        else:
            for provider, engine, config, filters in (
                (
                    "hip",
                    "HIP_MLOPS_ENGINE",
                    f"{source}/dnn-providers/hip-kernel-provider/config/HIP_MLOPS_ENGINE.toml",
                    "Smoke/IntegrationGpuRMSNormBackwardPureFp32.Correctness/0:Smoke/IntegrationGpuRMSNormBackwardPureFp16.Correctness/0:Smoke/IntegrationGpuRMSNormBackwardPureFp32.Correctness/1:Smoke/IntegrationGpuRMSNormBackwardMixedFp16.Correctness/0:Quick/IntegrationGpuLayernormBackwardPure4DFp32.Correctness/3:Quick/IntegrationGpuLayernormBackwardPure4DFp32.Correctness/10:Quick/IntegrationGpuLayernormBackwardMixed4DFp16.Correctness/3:Quick/IntegrationGpuLayernormBackwardMixed4DFp16.Correctness/10:Quick/IntegrationGpuLayernormBackwardMixed4DBfp16.Correctness/3:Quick/IntegrationGpuLayernormBackwardMixed4DBfp16.Correctness/10",
                ),
                (
                    "miopen",
                    "MIOPEN_ENGINE",
                    "",
                    "Smoke/IntegrationGpuBatchnormFwdTraining2dFp32.Correctness/0:Smoke/IntegrationGpuBatchnormFwdTraining2dFp32.Correctness/6:Smoke/IntegrationGpuBatchnormBackward2dFp32.Correctness/0:Smoke/IntegrationGpuBatchnormBackward2dFp32.Correctness/6",
                ),
            ):
                label = f"dnn-{provider}"
                plugin = (
                    "libhip_kernel_provider.so"
                    if provider == "hip"
                    else "libmiopen_plugin.so"
                )
                cmd = [
                    f"{prefix}/bin/hipdnn_integration_tests",
                    "--test-article",
                    f"{prefix}/lib/hipdnn_plugins/engines/{plugin}",
                    "--test-engine",
                    engine,
                    "--reference-executor",
                    "cpu",
                    "--verification-mode",
                    "cpu",
                    "--validator",
                    "cpu",
                    "--no-bundles",
                    "--fail-on-unsupported",
                    f"--gtest_filter={filters}",
                    f"--gtest_output=xml:/workspace/logs/{label}.xml",
                ]
                if config:
                    cmd += ["--test-config", config]
                run(label, cmd, xml=True, timeout=300)
        report["status"] = "passed"
    except BaseException:
        report["status"] = "failed"
        raise
    finally:
        output.write_text(json.dumps(report, indent=2) + "\n")
        with (logs / f"kernel-after-{args.group}.log").open("w") as stream:
            subprocess.run(["dmesg"], stdout=stream, check=True)


if __name__ == "__main__":
    main()
