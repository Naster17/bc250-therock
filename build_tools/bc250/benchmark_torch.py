# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Short, CPU-checked Torch benchmarks with matched inputs and synchronized timing."""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import threading
import time

import torch


def tensor_hash(tensors: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
        digest.update(
            tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def telemetry() -> dict:
    result = {"monotonic": time.monotonic()}
    roots = sorted(Path("/sys/class/drm/card0/device/hwmon").glob("hwmon*"))
    if roots:
        for name in ("freq1_input", "temp1_input", "power1_average"):
            path = roots[0] / name
            if path.exists():
                try:
                    result[name] = int(path.read_text().strip())
                except OSError:
                    pass
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--prepare-fixtures", action="store_true")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--sample-ms", type=float, default=100)
    parser.add_argument("--warmup-ms", type=float, default=400)
    parser.add_argument(
        "--cases",
        default="",
        help="comma-separated case names; empty runs the full short suite",
    )
    args = parser.parse_args()
    if not 3 <= args.samples <= 9 or not 50 <= args.sample_ms <= 250:
        parser.error("Keep this suite bounded: 3-9 samples of 50-250 ms")
    if args.prepare_fixtures:
        fixtures = {}
        for size in (1024, 2048):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                torch.manual_seed(1704 + size)
                fixtures[f"gemm_{size}_{str(dtype).split('.')[-1]}"] = {
                    "a": (torch.randn(size, size) / math.sqrt(size)).to(dtype),
                    "b": (torch.randn(size, size) / math.sqrt(size)).to(dtype),
                }
        for batch, channels in ((1, 16), (8, 32)):
            torch.manual_seed(1710 + channels)
            model = torch.nn.Sequential(
                torch.nn.Conv2d(channels, channels * 2, 3, padding=1),
                torch.nn.SiLU(),
                torch.nn.Conv2d(channels * 2, channels, 3, padding=1),
            )
            fixtures[f"conv_fwd_bwd_{batch}x{channels}x32x32_fp32"] = {
                "x": torch.randn(batch, channels, 32, 32),
                "weights": model.state_dict(),
            }
        args.fixtures.parent.mkdir(parents=True, exist_ok=True)
        torch.save(fixtures, args.fixtures)
        print("PERF_SHARED_FIXTURES_READY", flush=True)
        return
    fixtures = torch.load(args.fixtures, map_location="cpu", weights_only=True)
    with args.fixtures.open("rb") as stream:
        fixture_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    selected = set(args.cases.split(",")) if args.cases else None
    allowed = set(fixtures) | {"device_copy_128mib"}
    if selected and selected - allowed:
        parser.error(f"Unknown cases: {sorted(selected - allowed)}")
    if not 0 <= args.warmup_ms <= 2000:
        parser.error("Warmup must be bounded to 0-2000 ms")
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    device = torch.cuda.get_device_properties(0)
    assert device.gcnArchName.split(":")[0] == "gfx1013"
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    report = {
        "status": "running",
        "label": args.label,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "arch": device.gcnArchName,
        "wgp_count": device.multi_processor_count,
        "gpu_memory_bytes": device.total_memory,
        "kernel": platform.release(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "torch_threads": torch.get_num_threads(),
        "sdma": os.environ.get("HSA_ENABLE_SDMA"),
        "architecture_override": os.environ.get("HSA_OVERRIDE_GFX_VERSION"),
        "fixture_sha256": fixture_hash,
        "samples": args.samples,
        "sample_target_ms": args.sample_ms,
        "warmup_ms": args.warmup_ms,
        "cases": [],
        "telemetry": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stopped = threading.Event()

    def sample_telemetry() -> None:
        while not stopped.is_set():
            report["telemetry"].append(telemetry())
            stopped.wait(0.1)

    monitor = threading.Thread(target=sample_telemetry, daemon=True)
    monitor.start()

    def measure(function, iterations: int) -> tuple[float, float]:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        torch.cuda.synchronize()
        started = time.perf_counter()
        start.record()
        for _ in range(iterations):
            function()
        end.record()
        end.synchronize()
        wall_ms = (time.perf_counter() - started) * 1000
        return wall_ms / iterations, start.elapsed_time(end) / iterations

    def bench(
        name: str,
        function,
        inputs_sha256: str,
        first_ms: float,
        operations: int | None = None,
        bytes_per_call: int | None = None,
    ) -> None:
        started = time.monotonic()
        # The caller timed its first invocation before correctness checks.
        warm_deadline = time.monotonic() + args.warmup_ms / 1000
        while time.monotonic() < warm_deadline:
            for _ in range(16):
                function()
            torch.cuda.synchronize()
        trial_ms, _ = measure(function, 3)
        iterations = max(1, min(1024, math.ceil(args.sample_ms / max(trial_ms, 0.001))))
        wall, gpu = [], []
        for _ in range(args.samples):
            wall_ms, gpu_ms = measure(function, iterations)
            wall.append(wall_ms)
            gpu.append(gpu_ms)
        quartiles = statistics.quantiles(wall, n=4)
        median = statistics.median(wall)
        values = [
            item
            for item in report["telemetry"]
            if started <= item["monotonic"] <= time.monotonic()
        ]
        clocks = [item["freq1_input"] for item in values if "freq1_input" in item]
        temperatures = [item["temp1_input"] for item in values if "temp1_input" in item]
        case = {
            "name": name,
            "status": "passed",
            "inputs_sha256": inputs_sha256,
            "first_invocation_ms": first_ms,
            "iterations_per_sample": iterations,
            "wall_samples_ms": wall,
            "gpu_samples_ms": gpu,
            "wall_median_ms": median,
            "wall_iqr_percent": (quartiles[2] - quartiles[0]) / median * 100,
            "gpu_median_ms": statistics.median(gpu),
            "observed_gpu_clock_hz_median": (
                statistics.median(clocks) if clocks else None
            ),
            "peak_temperature_millicelsius": (
                max(temperatures) if temperatures else None
            ),
        }
        if operations:
            case["gflops"] = operations / (median * 1e6)
        if bytes_per_call:
            case["effective_gb_per_second"] = bytes_per_call / (median * 1e6)
        report["cases"].append(case)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"PERF_CASE {name} wall_ms={median:.5f} gpu_ms={case['gpu_median_ms']:.5f} iqr_pct={case['wall_iqr_percent']:.2f}",
            flush=True,
        )

    try:
        for size in (1024, 2048):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                name = f"gemm_{size}_{str(dtype).split('.')[-1]}"
                if selected and name not in selected:
                    continue
                a, b = fixtures[name]["a"], fixtures[name]["b"]
                inputs_hash = tensor_hash([a, b])
                reference = (a.double() @ b.double()).to(dtype)
                ga, gb = a.cuda(), b.cuda()
                torch.cuda.synchronize()
                cold_started = time.perf_counter()
                result = ga @ gb
                torch.cuda.synchronize()
                first_ms = (time.perf_counter() - cold_started) * 1000
                tolerance = {
                    torch.float32: (5e-4, 1e-5),
                    torch.float16: (3e-3, 1e-4),
                    torch.bfloat16: (2e-2, 1e-3),
                }[dtype]
                torch.testing.assert_close(
                    result.cpu(), reference, rtol=tolerance[0], atol=tolerance[1]
                )
                bench(
                    f"gemm_{size}_{str(dtype).split('.')[-1]}",
                    lambda: torch.mm(ga, gb),
                    inputs_hash,
                    first_ms,
                    operations=2 * size**3,
                )
                del a, b, ga, gb, result, reference
        for batch, channels in ((1, 16), (8, 32)):
            name = f"conv_fwd_bwd_{batch}x{channels}x32x32_fp32"
            if selected and name not in selected:
                continue
            torch.manual_seed(1710 + channels)
            cpu = torch.nn.Sequential(
                torch.nn.Conv2d(channels, channels * 2, 3, padding=1),
                torch.nn.SiLU(),
                torch.nn.Conv2d(channels * 2, channels, 3, padding=1),
            )
            cpu.load_state_dict(fixtures[name]["weights"])
            x = fixtures[name]["x"].clone().requires_grad_()
            inputs_hash = tensor_hash([x, *cpu.parameters()])
            expected_loss = cpu(x).square().mean()
            expected_loss.backward()
            model = copy.deepcopy(cpu).cuda()
            gx = x.detach().cuda().requires_grad_()

            def train_step():
                model.zero_grad(set_to_none=True)
                gx.grad = None
                loss = model(gx).square().mean()
                loss.backward()
                return loss

            torch.cuda.synchronize()
            cold_started = time.perf_counter()
            actual = train_step()
            torch.cuda.synchronize()
            first_ms = (time.perf_counter() - cold_started) * 1000
            torch.testing.assert_close(
                actual.cpu(), expected_loss.detach(), rtol=1e-4, atol=1e-5
            )
            torch.testing.assert_close(gx.grad.cpu(), x.grad, rtol=3e-3, atol=1e-5)
            for parameter, original in zip(model.parameters(), cpu.parameters()):
                torch.testing.assert_close(
                    parameter.grad.cpu(), original.grad, rtol=3e-3, atol=1e-5
                )
            bench(
                f"conv_fwd_bwd_{batch}x{channels}x32x32_fp32",
                train_step,
                inputs_hash,
                first_ms,
            )
            del cpu, model, x, gx, actual, expected_loss
        if not selected or "device_copy_128mib" in selected:
            source = torch.full((128 * 1024**2,), 173, device="cuda", dtype=torch.uint8)
            destination = torch.empty_like(source)
            torch.cuda.synchronize()
            cold_started = time.perf_counter()
            destination.copy_(source)
            torch.cuda.synchronize()
            first_ms = (time.perf_counter() - cold_started) * 1000
            assert torch.all(destination == 173).item()
            bench(
                "device_copy_128mib",
                lambda: destination.copy_(source),
                hashlib.sha256(b"uint8:134217728:173").hexdigest(),
                first_ms,
                bytes_per_call=source.numel() * 2,
            )
            assert torch.all(destination == 173).item()
        report["status"] = "passed"
    except BaseException:
        report["status"] = "failed"
        raise
    finally:
        stopped.set()
        monitor.join(timeout=2)
        report["telemetry_final"] = telemetry()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("TORCH_PERFORMANCE_SUITE_PASS", flush=True)


if __name__ == "__main__":
    main()
