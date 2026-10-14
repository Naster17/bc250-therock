#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Bounded CPU-reference hipDNN consumer qualification for gfx1013."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as et


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument(
        "--group", choices=("lt", "hip", "miopen", "fusion", "all"), default="all"
    )
    args = parser.parse_args()
    ws = args.workspace.resolve()
    logs = ws / "logs/priority1-dnn"
    logs.mkdir(parents=True, exist_ok=True)
    prefix = "/workspace/build/dist/rocm"
    providers = "/workspace/source/rocm-libraries/dnn-providers"
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
        f"{ws}:/workspace:rw",
        "-e",
        "HSA_ENABLE_SDMA=0",
        "-e",
        "OMP_NUM_THREADS=4",
        "-e",
        f"LD_LIBRARY_PATH={prefix}/lib:{prefix}/lib/llvm/lib",
        "localhost/bc250-rocm-build:10.2-ccache",
    ]
    definitions = {
        "lt": (
            "libhipblaslt_plugin.so",
            "HIPBLASLT_ENGINE",
            "hipblaslt-provider",
            [
                f"Smoke/IntegrationGpuMatmul{op}{dtype}.Correctness/{i}"
                for op in ("", "Bias", "Activ", "BiasActiv")
                for dtype in ("Fp32", "Fp16", "Bf16")
                for i in (0, 1, 2, 3)
            ],
        ),
        "hip": (
            "libhip_kernel_provider.so",
            "HIP_MLOPS_ENGINE",
            "hip-kernel-provider",
            [
                f"Smoke/IntegrationGpuRMSNormActivation{dtype}.Correctness/{i}"
                for dtype in ("PureFp32", "PureFp16", "MixedFp16", "MixedBfp16")
                for i in (0, 1, 2)
            ]
            + [
                f"Quick/IntegrationGpuLayernorm{dtype}.Correctness/{i}"
                # Mixed inference cases have an upstream CPU-reference skip; use training
                # cases here and retain that limitation rather than counting a skip as a pass.
                for dtype in ("Pure5DFp32", "Mixed5DFp16", "Mixed5DBfp16")
                for i in ((0, 3) if dtype == "Pure5DFp32" else (1, 3))
            ],
        ),
        "fusion": (
            "libmiopen_plugin.so",
            "MIOPEN_ENGINE",
            "miopen-provider",
            [
                f"Smoke/IntegrationGpuConvFwdBiasActiv{rank}dFp32.Correctness/{i}"
                for rank in (2, 3)
                for i in (0, 1, 30, 31)
            ]
            + [
                f"HomogeneousSmoke/IntegrationGpuConvFwdBiasActivHomogeneous2d{dtype}.Correctness/{i}"
                for dtype in ("Fp16", "Bfp16")
                for i in range(4)
            ],
        ),
        "miopen": (
            "libmiopen_plugin.so",
            "MIOPEN_ENGINE",
            "miopen-provider",
            [
                f"Smoke/IntegrationGpuConv{op}3d{dtype}.Correctness/{i}"
                for op in ("Fwd", "BwdData", "Wrw")
                for dtype in ("Fp32", "Fp16", "Bfp16")
                for i in (3, 12)
            ],
        ),
    }
    groups = definitions if args.group == "all" else [args.group]
    report = {"status": "running", "checks": []}
    output = logs / f"{args.group}.json"
    assert Path("/sys/class/watchdog/watchdog0/state").read_text().strip() == "active"
    with (logs / f"kernel-before-{args.group}.log").open("w") as stream:
        subprocess.run(["dmesg"], stdout=stream, check=True)
    try:
        for group in groups:
            plugin, engine, provider, filters = definitions[group]
            command = [
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
                "--test-config",
                f"{providers}/{provider}/config/{engine}.toml",
                "--gtest_filter=" + ":".join(filters),
                f"--gtest_output=xml:/workspace/logs/priority1-dnn/{group}.xml",
            ]
            start = time.monotonic()
            with (logs / f"{group}.log").open("w") as stream:
                result = subprocess.run(
                    base + ["timeout", "--kill-after=10s", "300s"] + command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=330,
                )
            assert result.returncode == 0, f'{group}: see {logs / (group + ".log")}'
            root = et.parse(logs / f"{group}.xml").getroot()
            cases = list(root.iter("testcase"))
            assert len(cases) == len(filters), (group, len(cases), len(filters))
            assert all(
                int(root.attrib.get(key, 0)) == 0
                for key in ("failures", "errors", "disabled")
            )
            assert all(
                c.attrib.get("result") == "completed" and c.find("skipped") is None
                for c in cases
            )
            report["checks"].append(
                {
                    "group": group,
                    "executed_tests": len(cases),
                    "seconds": time.monotonic() - start,
                }
            )
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
