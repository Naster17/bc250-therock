#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-checked bounded BC-250 backend and primitive timings."""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--gemm-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(250)
    torch.set_float32_matmul_precision("highest")
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1013"
    report = {
        "status": "running",
        "label": args.label,
        "torch": torch.__version__,
        "arch": torch.cuda.get_device_properties(0).gcnArchName,
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def bench(name, call, *, operations=0, traffic=0):
        for _ in range(3):
            call()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        a.record()
        call()
        b.record()
        b.synchronize()
        count = max(1, min(256, round(30 / max(a.elapsed_time(b), 0.001))))
        wall, device = [], []
        for _ in range(5):
            torch.cuda.synchronize()
            started = time.perf_counter()
            a.record()
            for _ in range(count):
                call()
            b.record()
            b.synchronize()
            wall.append((time.perf_counter() - started) * 1000 / count)
            device.append(a.elapsed_time(b) / count)
        item = {
            "name": name,
            "wall_samples_ms": wall,
            "gpu_samples_ms": device,
            "wall_median_ms": statistics.median(wall),
            "gpu_median_ms": statistics.median(device),
            "iterations": count,
        }
        if operations:
            item["gpu_gflops"] = operations / (item["gpu_median_ms"] * 1e6)
        if traffic:
            item["effective_gb_s"] = traffic / (item["gpu_median_ms"] * 1e6)
        report["cases"].append(item)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(name, item["gpu_median_ms"], flush=True)

    initial = torch.backends.cuda.preferred_blas_library()
    try:
        for m, n, k in (
            (32, 32, 32),
            (127, 65, 131),
            (256, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
            (64, 1024, 256),
        ):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                a = (torch.randn(m, k) / k**0.5).to(dtype)
                b = (torch.randn(k, n) / k**0.5).to(dtype)
                expected = (a.double() @ b.double()).to(dtype)
                ga, gb = a.cuda(), b.cuda()
                tol = {
                    torch.float32: (5e-4, 2e-6),
                    torch.float16: (3e-3, 1e-4),
                    torch.bfloat16: (2e-2, 1e-3),
                }[dtype]
                for backend in ("cublas", "hipblaslt"):
                    torch.backends.cuda.preferred_blas_library(backend)
                    result = ga @ gb
                    torch.testing.assert_close(
                        result.cpu(), expected, rtol=tol[0], atol=tol[1]
                    )
                    bench(
                        f"gemm_{m}x{n}x{k}_{dtype}_{backend}",
                        lambda: torch.mm(ga, gb),
                        operations=2 * m * n * k,
                    )
        torch.backends.cuda.preferred_blas_library(initial)
        if not args.gemm_only:
            for size in (257, 65537, 1048576):
                cpu = torch.randn(size)
                gpu = cpu.cuda()
                for op in ("sum", "dot"):
                    call = (
                        (lambda: gpu.sum())
                        if op == "sum"
                        else (lambda: torch.dot(gpu, gpu))
                    )
                    expected = (
                        cpu.double().sum()
                        if op == "sum"
                        else cpu.double().square().sum()
                    )
                    torch.testing.assert_close(
                        call().cpu().double(), expected, rtol=3e-5, atol=3e-4
                    )
                    bench(f"{op}_{size}", call, traffic=size * 4)
            # Contrast generated reductions/fusion with eager launch sequences;
            # compile cost is excluded from steady timing and checked separately.
            for size in (65537, 1048576):
                cpu = torch.randn(size)
                gpu = cpu.cuda()
                for name, function in (
                    ("sum", lambda value: value.sum()),
                    ("dot", lambda value: torch.dot(value, value)),
                    (
                        "fused_silu_square_sum",
                        lambda value: torch.nn.functional.silu(value).square().sum(),
                    ),
                ):
                    expected = function(cpu.double())
                    for mode in ("eager", "compiled"):
                        call = (
                            function
                            if mode == "eager"
                            else torch.compile(function, fullgraph=True)
                        )
                        result = call(gpu)
                        torch.testing.assert_close(
                            result.cpu().double(), expected, rtol=3e-5, atol=3e-4
                        )
                        bench(f"{name}_{size}_{mode}", lambda: call(gpu))
                    graph = torch.cuda.CUDAGraph()
                    torch.cuda.synchronize()
                    with torch.cuda.graph(graph):
                        captured = call(gpu)
                    graph.replay()
                    torch.testing.assert_close(
                        captured.cpu().double(), expected, rtol=3e-5, atol=3e-4
                    )
                    bench(f"{name}_{size}_compiled_hipgraph", graph.replay)
                    torch.testing.assert_close(
                        captured.cpu().double(), expected, rtol=3e-5, atol=3e-4
                    )

            cpu = torch.arange(8 * 1024 * 1024, dtype=torch.float32)
            source = cpu.cuda()
            target = torch.empty_like(source)
            target.copy_(source)
            torch.testing.assert_close(target.cpu(), cpu)
            bench("copy_32mib", lambda: target.copy_(source), traffic=cpu.numel() * 8)
            tiny = torch.randn(1, device="cuda")
            out = torch.empty_like(tiny)
            bench("launch_add_scalar", lambda: torch.add(tiny, 1, out=out))
            for size in (32, 128, 256):
                base = torch.randn(size, size, dtype=torch.float64)
                a = base @ base.T + size * torch.eye(size, dtype=torch.float64)
                b = torch.randn(size, 4, dtype=torch.float64)
                ga, gb = a.float().cuda(), b.float().cuda()
                expected = torch.linalg.solve(a, b)
                torch.testing.assert_close(
                    torch.linalg.solve(ga, gb).cpu().double(),
                    expected,
                    rtol=4e-4,
                    atol=2e-6,
                )
                bench(f"solve_{size}", lambda: torch.linalg.solve(ga, gb))
                l = torch.linalg.cholesky(ga)
                torch.testing.assert_close(
                    torch.cholesky_solve(gb, l).cpu().double(),
                    expected,
                    rtol=4e-4,
                    atol=2e-6,
                )
                bench(f"cholesky_factor_{size}", lambda: torch.linalg.cholesky(ga))
                bench(f"cholesky_reused_{size}", lambda: torch.cholesky_solve(gb, l))
        report["status"] = "passed"
    except BaseException:
        report["status"] = "failed"
        raise
    finally:
        torch.backends.cuda.preferred_blas_library(initial)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("PRIORITY2_PERFORMANCE_PASS", flush=True)


if __name__ == "__main__":
    main()
