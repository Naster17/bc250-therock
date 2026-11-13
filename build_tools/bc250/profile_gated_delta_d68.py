#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""D68 native gated-delta solve/scan cost attribution (measurement only).

D66 closed the launch-bound frozen-loop gap; A06 recorded that the remaining
ATen solve/scan dominates the native chunk_gated_delta_rule at length (~7%
native win at B1/H16/D128/L1024). This owned fixture attributes the native op's
cost before any fusion: full forward+backward baseline, forward CUDA-event
section breakdown on an exact replica of the installed op body, and isolated
forward+backward microbenchmarks of the two batched triangular solves and the
sequential chunk scan. The replica is verified against the installed op output
before its breakdown is trusted.

Owned-only: random inputs, no models, no user data, nothing installed. Runs
under the ordinary bc250-train caller on the immutable release. Measurement,
not a speed claim; the fusion scope is chosen from these numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

FIXTURE = "profile_gated_delta_d68"
DTYPES = ("float32", "bfloat16", "float16")
# label, batch, seq, heads, dk, dv
GEOMETRIES = (
    ("a06-long", 1, 1024, 16, 128, 128),
    ("a06-short", 1, 65, 16, 128, 128),
    ("d66-benchmark", 2, 32, 2, 16, 16),
    ("d66-long256", 2, 256, 2, 16, 16),
)
WARMUP = 3
ROUNDS = 6
REPLICA_REL_L2_BOUND = 1e-6


# ---------------------------------------------------------------------------
# Pure logic (no torch import; unit-testable on the workstation).
# ---------------------------------------------------------------------------
def new_report() -> dict:
    return {
        "fixture": FIXTURE,
        "status": "running",
        "protocol": {
            "warmup": WARMUP,
            "rounds": ROUNDS,
            "replica_rel_l2_bound": REPLICA_REL_L2_BOUND,
            "note": "wall time with per-call sync for whole-op/micro benchmarks; "
            "async CUDA events read after step sync for the forward breakdown",
        },
        "geometries": [
            {"label": label, "batch": b, "seq": t, "heads": h, "dk": dk, "dv": dv}
            for label, b, t, h, dk, dv in GEOMETRIES
        ],
        "versions": {},
        "module": {},
        "sections": {},
    }


def summarize(samples: list[float]) -> dict:
    ordered = sorted(samples)
    n = len(ordered)
    med = statistics.median(ordered)
    if n < 2 or med <= 0:
        iqr = 0.0
    else:
        q1 = statistics.median(ordered[: n // 2])
        q3 = statistics.median(ordered[(n + 1) // 2 :])
        iqr = (q3 - q1) / med * 100.0
    return {"samples_ms": samples, "median_ms": med, "iqr_percent": iqr}


def fraction(part_ms: float, whole_ms: float) -> float:
    return part_ms / whole_ms if whole_ms > 0 else 0.0


def chunk_count(seq: int) -> int:
    return (seq + 63) // 64


# ---------------------------------------------------------------------------
# Torch-dependent sections (run on the board under bc250-train).
# ---------------------------------------------------------------------------
def _inputs(torch, dtype, batch, seq, heads, dk, dv, device):
    compute = getattr(torch, dtype)
    torch.manual_seed(6801)
    q = torch.randn(batch, seq, heads, dk, device=device, dtype=compute) * 0.15
    k = torch.randn(batch, seq, heads, dk, device=device, dtype=compute) * 0.15
    v = torch.randn(batch, seq, heads, dv, device=device, dtype=compute) * 0.2
    g = -(torch.rand(batch, seq, heads, device=device) * 0.15 + 0.01)
    beta = torch.rand(batch, seq, heads, device=device) * 0.8 + 0.1
    return q, k, v, g, beta


def _rel_l2(actual, reference) -> float:
    a = actual.detach().double()
    b = reference.detach().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def _op_call(ops, xs, **kwargs):
    return ops.chunk_gated_delta_rule(
        *xs, use_qk_l2norm_in_kernel=True, output_final_state=True, **kwargs
    )


def _op_loss(ops, xs):
    import torch

    params = [x.detach().clone().requires_grad_(True) for x in xs]
    out, state = _op_call(ops, params)
    loss = out.square().sum() + state.square().sum()
    grads = torch.autograd.grad(loss, params)
    return out, state, grads


def _timed(fn, warmup=WARMUP, rounds=ROUNDS):
    for _ in range(warmup):
        fn()
    import torch

    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return summarize(samples)


def _replica_sections(torch, ops, xs):
    """Exact replica of the installed chunk_gated_delta_rule body with async
    CUDA-event pairs around each section. Returns (out, state, elapsed_ms)."""
    query, key, value, g, beta = xs
    prepare = (
        ops.prepare_gated_delta
        if query.dtype == torch.float32 and not torch.is_autocast_enabled("cuda")
        else ops._reference_ordered_prepare
    )
    marks = {}

    def mark(name):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        marks[name] = e

    mark("start")
    q, k, kb, vb, decay = prepare(query, key, value, g, beta, normalize=True)
    mark("prepared")
    b, t, h, dk = query.shape
    dv = value.shape[-1]
    mask = torch.ones((64, 64), device=query.device, dtype=torch.bool).triu(1)
    pair = (
        (decay.unsqueeze(-1) - decay.unsqueeze(-2))
        .masked_fill(mask, -float("inf"))
        .exp()
    )
    system = (kb @ k.transpose(-1, -2)) * pair
    attention = (q @ k.transpose(-1, -2)) * pair
    decayed_k_beta = kb * decay.exp().unsqueeze(-1)
    mark("gemms")
    u = torch.linalg.solve_triangular(system, vb, upper=False, unitriangular=True)
    w = torch.linalg.solve_triangular(
        system, decayed_k_beta, upper=False, unitriangular=True
    )
    mark("solves")
    q = q * decay.exp().unsqueeze(-1)
    k = k * (decay[..., -1:] - decay).exp().unsqueeze(-1)
    chunk_decay = decay[..., -1].exp()[..., None, None]
    state = torch.zeros((b, h, dk, dv), device=query.device, dtype=torch.float32)
    out = torch.zeros_like(u)
    for i in range(q.shape[2]):
        delta = u[:, :, i] - w[:, :, i] @ state
        inter_chunk = q[:, :, i] @ state
        out[:, :, i] = inter_chunk + attention[:, :, i] @ delta
        state = state * chunk_decay[:, :, i] + k[:, :, i].transpose(-1, -2) @ delta
    mark("scan")
    out = out.flatten(2, 3)[:, :, :t].transpose(1, 2)
    result = out.to(query.dtype).contiguous()
    mark("end")
    torch.cuda.synchronize()
    order = ["start", "prepared", "gemms", "solves", "scan", "end"]
    elapsed = {
        f"{a}->{b}": marks[a].elapsed_time(marks[b]) for a, b in zip(order, order[1:])
    }
    return result, state, elapsed


def section_baseline(report, ops, device):
    import torch

    section = {"status": "running", "cases": []}
    report["sections"]["baseline"] = section
    for label, b, t, h, dk, dv in GEOMETRIES:
        for dtype in DTYPES:
            entry = {
                "geometry": label,
                "dtype": dtype,
                "chunks": chunk_count(t),
                "status": "running",
            }
            section["cases"].append(entry)
            try:
                xs = _inputs(torch, dtype, b, t, h, dk, dv, device)
                entry["full_fwd_bwd"] = _timed(lambda: _op_loss(ops, xs))
                entry["status"] = "passed"
                print(
                    f"D68_BASELINE {label} {dtype} "
                    f"fwd_bwd={entry['full_fwd_bwd']['median_ms']:.3f}ms "
                    f"iqr={entry['full_fwd_bwd']['iqr_percent']:.2f}%",
                    flush=True,
                )
            except Exception as exc:
                entry["status"] = "failed"
                entry["error"] = f"{type(exc).__name__}: {exc}"
    section["status"] = (
        "passed" if all(c["status"] == "passed" for c in section["cases"]) else "failed"
    )


def section_breakdown(report, ops, device):
    import torch

    section = {"status": "running", "cases": []}
    report["sections"]["breakdown"] = section
    for label, b, t, h, dk, dv in GEOMETRIES:
        for dtype in DTYPES:
            entry = {"geometry": label, "dtype": dtype, "status": "running"}
            section["cases"].append(entry)
            try:
                xs = _inputs(torch, dtype, b, t, h, dk, dv, device)
                ref_out, ref_state, _ = _op_loss(ops, xs)
                for _ in range(WARMUP):
                    _replica_sections(torch, ops, xs)
                runs = []
                for _ in range(ROUNDS):
                    out, state, elapsed = _replica_sections(torch, ops, xs)
                    runs.append(elapsed)
                replica_rel = max(_rel_l2(out, ref_out), _rel_l2(state, ref_state))
                entry["replica_rel_l2"] = replica_rel
                if replica_rel > REPLICA_REL_L2_BOUND:
                    raise ValueError(
                        f"replica diverges from installed op: {replica_rel:.3e}"
                    )
                keys = list(runs[0])
                entry["sections_ms"] = {
                    key: summarize([r[key] for r in runs]) for key in keys
                }
                entry["status"] = "passed"
                parts = " ".join(
                    f"{key}={entry['sections_ms'][key]['median_ms']:.3f}"
                    for key in keys
                )
                print(f"D68_BREAKDOWN {label} {dtype} {parts}", flush=True)
            except Exception as exc:
                entry["status"] = "failed"
                entry["error"] = f"{type(exc).__name__}: {exc}"
    section["status"] = (
        "passed" if all(c["status"] == "passed" for c in section["cases"]) else "failed"
    )


def _solve_micro(torch, ops, xs):
    """Isolated cost of the two batched unit-triangular solves, fwd+bwd."""
    query, key, value, g, beta = xs
    prepare = (
        ops.prepare_gated_delta
        if query.dtype == torch.float32 and not torch.is_autocast_enabled("cuda")
        else ops._reference_ordered_prepare
    )
    q, k, kb, vb, decay = prepare(query, key, value, g, beta, normalize=True)
    mask = torch.ones((64, 64), device=query.device, dtype=torch.bool).triu(1)
    pair = (
        (decay.unsqueeze(-1) - decay.unsqueeze(-2))
        .masked_fill(mask, -float("inf"))
        .exp()
    )
    system = ((kb @ k.transpose(-1, -2)) * pair).detach().requires_grad_(True)
    vb = vb.detach().requires_grad_(True)
    decayed_k_beta = (kb * decay.exp().unsqueeze(-1)).detach().requires_grad_(True)

    def run():
        u = torch.linalg.solve_triangular(system, vb, upper=False, unitriangular=True)
        w = torch.linalg.solve_triangular(
            system, decayed_k_beta, upper=False, unitriangular=True
        )
        loss = u.square().sum() + w.square().sum()
        torch.autograd.grad(loss, (system, vb, decayed_k_beta))

    return run


def _scan_micro(torch, ops, xs):
    """Isolated cost of the sequential chunk scan, fwd+bwd."""
    query, key, value, g, beta = xs
    prepare = (
        ops.prepare_gated_delta
        if query.dtype == torch.float32 and not torch.is_autocast_enabled("cuda")
        else ops._reference_ordered_prepare
    )
    q, k, kb, vb, decay = prepare(query, key, value, g, beta, normalize=True)
    b, t, h, dk = query.shape
    dv = value.shape[-1]
    mask = torch.ones((64, 64), device=query.device, dtype=torch.bool).triu(1)
    pair = (
        (decay.unsqueeze(-1) - decay.unsqueeze(-2))
        .masked_fill(mask, -float("inf"))
        .exp()
    )
    system = (kb @ k.transpose(-1, -2)) * pair
    attention = ((q @ k.transpose(-1, -2)) * pair).detach().requires_grad_(True)
    u = (
        torch.linalg.solve_triangular(system, vb, upper=False, unitriangular=True)
        .detach()
        .requires_grad_(True)
    )
    w = (
        torch.linalg.solve_triangular(
            system, kb * decay.exp().unsqueeze(-1), upper=False, unitriangular=True
        )
        .detach()
        .requires_grad_(True)
    )
    q = (q * decay.exp().unsqueeze(-1)).detach().requires_grad_(True)
    k = (
        (k * (decay[..., -1:] - decay).exp().unsqueeze(-1))
        .detach()
        .requires_grad_(True)
    )
    chunk_decay = decay[..., -1].exp()[..., None, None].detach().requires_grad_(True)
    leaves = (q, k, u, w, attention, chunk_decay)

    def run():
        state = torch.zeros((b, h, dk, dv), device=query.device, dtype=torch.float32)
        out = torch.zeros_like(u)
        for i in range(q.shape[2]):
            delta = u[:, :, i] - w[:, :, i] @ state
            inter_chunk = q[:, :, i] @ state
            out[:, :, i] = inter_chunk + attention[:, :, i] @ delta
            state = state * chunk_decay[:, :, i] + k[:, :, i].transpose(-1, -2) @ delta
        loss = out.square().sum() + state.square().sum()
        torch.autograd.grad(loss, leaves)

    return run


def section_micro(report, ops, device):
    import torch

    section = {"status": "running", "cases": []}
    report["sections"]["micro"] = section
    for label, b, t, h, dk, dv in GEOMETRIES:
        for dtype in DTYPES:
            entry = {"geometry": label, "dtype": dtype, "status": "running"}
            section["cases"].append(entry)
            try:
                xs = _inputs(torch, dtype, b, t, h, dk, dv, device)
                entry["solves_fwd_bwd"] = _timed(_solve_micro(torch, ops, xs))
                entry["scan_fwd_bwd"] = _timed(_scan_micro(torch, ops, xs))
                entry["status"] = "passed"
                print(
                    f"D68_MICRO {label} {dtype} "
                    f"solves={entry['solves_fwd_bwd']['median_ms']:.3f}ms "
                    f"scan={entry['scan_fwd_bwd']['median_ms']:.3f}ms",
                    flush=True,
                )
            except Exception as exc:
                entry["status"] = "failed"
                entry["error"] = f"{type(exc).__name__}: {exc}"
    section["status"] = (
        "passed" if all(c["status"] == "passed" for c in section["cases"]) else "failed"
    )


def section_summary(report):
    """Attribute the measured solve+scan share of the full op per case."""
    section = {"status": "running", "cases": []}
    report["sections"]["summary"] = section
    baseline = {
        (c["geometry"], c["dtype"]): c
        for c in report["sections"]["baseline"]["cases"]
        if c["status"] == "passed"
    }
    micro = {
        (c["geometry"], c["dtype"]): c
        for c in report["sections"]["micro"]["cases"]
        if c["status"] == "passed"
    }
    for key, base in baseline.items():
        if key not in micro:
            continue
        m = micro[key]
        whole = base["full_fwd_bwd"]["median_ms"]
        solves = m["solves_fwd_bwd"]["median_ms"]
        scan = m["scan_fwd_bwd"]["median_ms"]
        entry = {
            "geometry": key[0],
            "dtype": key[1],
            "full_fwd_bwd_ms": whole,
            "solves_fwd_bwd_ms": solves,
            "scan_fwd_bwd_ms": scan,
            "solves_fraction": fraction(solves, whole),
            "scan_fraction": fraction(scan, whole),
            "combined_fraction": fraction(solves + scan, whole),
        }
        section["cases"].append(entry)
        print(
            f"D68_SUMMARY {key[0]} {key[1]} full={whole:.3f}ms "
            f"solves={solves:.3f}ms ({entry['solves_fraction']:.1%}) "
            f"scan={scan:.3f}ms ({entry['scan_fraction']:.1%}) "
            f"combined={entry['combined_fraction']:.1%}",
            flush=True,
        )
    section["status"] = "passed" if section["cases"] else "failed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import torch
    import triton
    from torch.nn import bc250_gated_delta as ops

    report = new_report()
    report["versions"] = {
        "torch": torch.__version__,
        "triton": triton.__version__,
        "python": __import__("sys").version.split()[0],
    }
    report["module"] = {
        "path": ops.__file__,
        "sha256": hashlib.sha256(Path(ops.__file__).read_bytes()).hexdigest(),
    }
    props = torch.cuda.get_device_properties(args.device)
    report["device"] = {"name": props.name, "gcnArchName": props.gcnArchName}
    print(
        f"D68 module={report['module']['sha256'][:12]} "
        f"torch={torch.__version__} triton={triton.__version__}",
        flush=True,
    )
    for fn in (section_baseline, section_breakdown, section_micro):
        fn(report, ops, args.device)
    section_summary(report)
    failed = [
        name
        for name, section in report["sections"].items()
        if section["status"] != "passed"
    ]
    report["status"] = "passed" if not failed else "failed"
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(f"PROFILE_D68 {report['status']} failed_sections={failed}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
