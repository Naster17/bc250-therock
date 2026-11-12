#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PLAN-10 owned training-performance benchmark (D64).

Bounded steady-state training-step timing for owned random-init models on the
native gfx1013 provider. Reuses the D62 owned model registry/builders and the
qualified timing methodology: lr=0 holds weights fixed during timing (optimizer
state still updates), a cold first step, warmup, then repeated synchronized
intervals summarized by median/IQR, tokens/s, peak memory and numerical checks.

Owned-only: no downloaded weights, no pretrained loads, no excluded applications.
Direct GPU utilization is unavailable on this board (gpu_busy_percent returns
EOPNOTSUPP); the report records that boundary plus hwmon temperature and the DPM
governor as context. This is a bounded measurement, not a production-speed,
model-quality or general two-board speedup claim.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import admit_model_d62 as admission  # noqa: E402  (owned registry/builders)

FIXTURE = "benchmark_training_d64"
BATCH = 2
SEQ = 64
WARMUP = 3
INTERVALS = 5
STEPS_PER_INTERVAL = 3
DTYPES = ("float32", "bfloat16", "float16")


# ---------------------------------------------------------------------------
# Pure logic (no torch import; unit-testable on the workstation).
# ---------------------------------------------------------------------------
def case_matrix() -> list[tuple[str, str, str]]:
    """(model, dtype, backend) cases over each model's qualified backends."""
    cases = []
    for name, spec in admission.MODEL_REGISTRY.items():
        for backend in spec["backends"]:
            for dtype in DTYPES:
                cases.append((name, dtype, backend))
    return cases


def tokens_per_second(token_count: int, step_median_ms: float) -> float:
    return token_count * 1000.0 / step_median_ms


def iqr_percent(samples: list[float]) -> float:
    median = statistics.median(samples)
    quartiles = statistics.quantiles(samples, n=4)
    return (quartiles[2] - quartiles[0]) / median * 100.0


def new_report() -> dict:
    return {
        "fixture": FIXTURE,
        "status": "running",
        "workload": {
            "batch": BATCH,
            "sequence_length": SEQ,
            "optimizer": "AdamW",
            "lr": 0,
        },
        "warmup_steps": WARMUP,
        "intervals": INTERVALS,
        "steps_per_interval": STEPS_PER_INTERVAL,
        "versions": {},
        "utilization": {},
        "cases": [],
    }


def summarize_case(samples: list[float], token_count: int) -> dict:
    median = statistics.median(samples)
    return {
        "step_samples_ms": samples,
        "step_median_ms": median,
        "step_iqr_percent": iqr_percent(samples),
        "tokens_per_second": tokens_per_second(token_count, median),
    }


def read_utilization() -> dict:
    """Best-effort GPU utilization/thermal context; honest about the boundary."""
    info: dict = {"gpu_busy_percent": None, "note": ""}
    busy = Path("/sys/class/drm/card0/device/gpu_busy_percent")
    try:
        info["gpu_busy_percent"] = int(busy.read_text().strip())
    except (OSError, ValueError):
        info["note"] = "gpu_busy_percent unavailable (EOPNOTSUPP on this board)"
    try:
        hwmon = Path("/sys/class/drm/card0/device/hwmon")
        temp = next(hwmon.glob("hwmon*/temp1_input"))
        info["temperature_c"] = int(temp.read_text().strip()) / 1000.0
    except (OSError, StopIteration, ValueError):
        pass
    try:
        gov = Path("/sys/class/drm/card0/device/power_dpm_force_performance_level")
        info["governor"] = gov.read_text().strip()
    except OSError:
        pass
    return info


# ---------------------------------------------------------------------------
# Torch-dependent benchmark (lazy imports; runs on the board under the runtime).
# ---------------------------------------------------------------------------
def benchmark_case(
    report: dict, name: str, dtype: str, backend: str, device: str
) -> None:
    import torch

    case = {"model": name, "dtype": dtype, "backend": backend, "status": "running"}
    report["cases"].append(case)
    try:
        torch.manual_seed(1704)
        model = admission.build_model(name, dtype, backend, device)
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=0)
        compute_dtype = getattr(torch, dtype)
        scaler = torch.amp.GradScaler(
            "cuda", enabled=(dtype == "float16"), init_scale=128
        )
        vocab = admission.registry_spec(name)["config"]["vocab_size"]
        ids = torch.randint(0, vocab, (BATCH, SEQ), device=device)
        token_count = ids.numel()

        def step():
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                "cuda", dtype=compute_dtype, enabled=(dtype != "float32")
            ):
                loss = model(input_ids=ids, labels=ids, use_cache=False).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 1)
            scaler.step(optimizer)
            scaler.update()
            return loss

        first_started = time.perf_counter()
        first_loss = step()
        torch.cuda.synchronize()
        first_ms = (time.perf_counter() - first_started) * 1000.0
        if not torch.isfinite(first_loss).item():
            raise ValueError("non-finite first loss")
        initial_scale = scaler.get_scale()
        for _ in range(WARMUP):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(INTERVALS):
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(STEPS_PER_INTERVAL):
                loss = step()
            torch.cuda.synchronize()
            samples.append(
                (time.perf_counter() - started) * 1000.0 / STEPS_PER_INTERVAL
            )
        if not torch.isfinite(loss).item():
            raise ValueError("non-finite timed loss")
        if scaler.get_scale() < initial_scale:
            raise ValueError("AMP skipped an update due to overflow")
        torch.testing.assert_close(loss, first_loss, rtol=1e-4, atol=1e-4)
        case.update(summarize_case(samples, token_count))
        case.update(
            {
                "status": "passed",
                "first_step_ms": first_ms,
                "trainable_parameters": sum(p.numel() for p in parameters),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "tokens_per_interval": token_count,
            }
        )
        del model, optimizer, scaler, parameters, first_loss, loss
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as exc:
        case["status"] = "failed"
        case["error"] = f"{type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="restrict to a subset of registry models",
    )
    parser.add_argument("--dtypes", nargs="*", default=list(DTYPES), choices=DTYPES)
    args = parser.parse_args()

    violations = admission.owned_import_guard(sys.path)
    if violations:
        raise SystemExit(
            f"Owned-import guard refused excluded sys.path entries: {violations}"
        )

    report = new_report()
    report["versions"] = admission.record_versions()
    report["utilization"] = read_utilization()
    cases = case_matrix()
    if args.models:
        cases = [c for c in cases if c[0] in args.models]
    cases = [c for c in cases if c[1] in args.dtypes]
    for name, dtype, backend in cases:
        benchmark_case(report, name, dtype, backend, args.device)
        latest = report["cases"][-1]
        if latest["status"] == "passed":
            print(
                f"BENCH_D64 {name} {dtype} {backend} "
                f"step_ms={latest['step_median_ms']:.3f} "
                f"tokens_s={latest['tokens_per_second']:.1f} "
                f"iqr_pct={latest['step_iqr_percent']:.2f}",
                flush=True,
            )
        else:
            print(
                f"BENCH_D64 {name} {dtype} {backend} FAILED "
                f"{latest.get('error', '')}",
                flush=True,
            )
    passed = sum(c["status"] == "passed" for c in report["cases"])
    report["status"] = (
        "passed" if passed == len(report["cases"]) and passed else "failed"
    )
    report["passed_cases"] = passed
    report["total_cases"] = len(report["cases"])
    report["fixture_sha256"] = admission.sha256_bytes(Path(__file__).read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"BENCHMARK_D64 {report['status']} {passed}/{len(report['cases'])} cases",
        flush=True,
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
