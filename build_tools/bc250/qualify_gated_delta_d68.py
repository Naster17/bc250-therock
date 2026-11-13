#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""D68 fused solve/scan qualification for the native gated-delta chunk op.

Qualifies the candidate module's opt-in use_fused_solve_scan path against the
default ATen path, an independent CPU/FP64 token recurrence, and the frozen
Transformers 5.2.0 hybrid model through the D66 routing helper (with the fused
flag injected). The candidate is loaded from BC250_D68_CANDIDATE; the installed
release module is imported normally and is never modified. The default path of
the candidate must remain bitwise identical to the installed module.

Pre-declared bounds (D68, set before any run; no post-hoc relaxation):
- op fused-vs-default rel-L2: FP32 1e-5, BF16/FP16 1e-2 (the D66 op bounds);
- op CPU/FP64 recurrence anchor: the A06 recurrent tolerances per dtype;
- model fused-vs-frozen: the D66 revised model bounds + CPU/FP64 anchor 5e-3.

The candidate applies the fused path only at >= FUSED_SCAN_MIN_CHUNKS chunks
(measured D68 crossover; single-chunk calls keep the default path), so the
d66-benchmark geometry is expected to time at parity, not faster.

Owned-only: random inputs and registry models, no downloaded weights, no user
data, nothing installed. Bounded qualification, not a production-speed or
model-quality claim.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import admit_model_d62 as admission  # noqa: E402
import benchmark_training_d64 as benchmark  # noqa: E402
import route_qwen35_gated_delta_d66 as routing  # noqa: E402

FIXTURE = "qualify_gated_delta_d68"
MODEL = "qwen35-hybrid-tiny"
BACKEND = "sdpa"
DTYPES = ("float32", "bfloat16", "float16")
# label, batch, seq, heads, dk, dv
GEOMETRIES = (
    ("a06-long", 1, 1024, 16, 128, 128),
    ("a06-short", 1, 65, 16, 128, 128),
    ("d66-benchmark", 2, 32, 2, 16, 16),
    ("d66-long256", 2, 256, 2, 16, 16),
)
OP_WARMUP = 2
OP_ROUNDS = 6

OP_REL_L2 = {"float32": 1e-5, "bfloat16": 1e-2, "float16": 1e-2}
# A06 recurrent tolerances (atol, rtol) against the CPU FP64 token recurrence.
CPU_RECURRENCE_TOL = {
    "float32": (2e-5, 2e-4),
    "float16": (0.001, 0.008),
    "bfloat16": (0.006, 0.04),
}
MODEL_FP32_LOSS_ATOL = 1e-4
MODEL_FP32_LOGITS_REL_L2 = 1e-4
MODEL_FP32_GRAD_REL_L2 = 2e-3
CPU_ANCHOR_GRAD_REL_L2 = 5e-3
MODEL_LOW_LOSS_ATOL = 0.015
MODEL_LOW_GRAD_REL_L2 = 0.04


# ---------------------------------------------------------------------------
# Pure logic (no torch import; unit-testable on the workstation).
# ---------------------------------------------------------------------------
def new_report() -> dict:
    return {
        "fixture": FIXTURE,
        "status": "running",
        "parity_bounds": {
            "op_rel_l2": OP_REL_L2,
            "cpu_recurrence_tol": CPU_RECURRENCE_TOL,
            "model_fp32_loss_atol": MODEL_FP32_LOSS_ATOL,
            "model_fp32_logits_rel_l2": MODEL_FP32_LOGITS_REL_L2,
            "model_fp32_grad_rel_l2": MODEL_FP32_GRAD_REL_L2,
            "cpu_anchor_grad_rel_l2": CPU_ANCHOR_GRAD_REL_L2,
            "model_low_loss_atol": MODEL_LOW_LOSS_ATOL,
            "model_low_grad_rel_l2": MODEL_LOW_GRAD_REL_L2,
        },
        "geometries": [
            {"label": label, "batch": b, "seq": t, "heads": h, "dk": dk, "dv": dv}
            for label, b, t, h, dk, dv in GEOMETRIES
        ],
        "versions": {},
        "modules": {},
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


def speedup(before_ms: float, after_ms: float) -> float:
    return before_ms / after_ms if after_ms > 0 else 0.0


def load_candidate(path: str):
    spec = importlib.util.spec_from_file_location("gated_delta_d68_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Torch-dependent sections (run on the board under bc250-train).
# ---------------------------------------------------------------------------
def _rel_l2(actual, reference) -> float:
    a = actual.detach().double()
    b = reference.detach().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def _inputs(torch, dtype, batch, seq, heads, dk, dv, device, seed=6801):
    compute = getattr(torch, dtype)
    torch.manual_seed(seed)
    q = torch.randn(batch, seq, heads, dk, device=device, dtype=compute) * 0.15
    k = torch.randn(batch, seq, heads, dk, device=device, dtype=compute) * 0.15
    v = torch.randn(batch, seq, heads, dv, device=device, dtype=compute) * 0.2
    g = -(torch.rand(batch, seq, heads, device=device) * 0.15 + 0.01)
    beta = torch.rand(batch, seq, heads, device=device) * 0.8 + 0.1
    return q, k, v, g, beta


def _op_grads(torch, ops, xs, fused, state=None):
    params = [x.detach().clone().requires_grad_(True) for x in xs]
    s0 = None if state is None else state.detach().clone().requires_grad_(True)
    kwargs = {} if fused is None else {"use_fused_solve_scan": fused}
    out, final = ops.chunk_gated_delta_rule(
        *params,
        initial_state=s0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        **kwargs,
    )
    leaves = (*params, s0) if s0 is not None else params
    loss = out.square().sum() + final.square().sum()
    grads = torch.autograd.grad(loss, leaves)
    return out, final, grads


def _recurrent_reference(torch, xs, normalize, initial=None):
    """CPU FP64 token-by-token recurrence (A06 independent oracle).

    Uses the passed tensors directly so autograd flows when they require grad.
    """
    q, k, v, g, beta = xs
    if normalize:
        q = q * (q.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
        k = k * (k.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
    q = q * q.shape[-1] ** -0.5
    b, t, h, dk = q.shape
    state = (
        torch.zeros((b, h, dk, v.shape[-1]), dtype=q.dtype)
        if initial is None
        else initial
    )
    outputs = []
    for i in range(t):
        state = state * g[:, i].exp()[..., None, None]
        error = v[:, i] - (k[:, i, :, :, None] * state).sum(-2)
        state = (
            state + k[:, i, :, :, None] * (beta[:, i, :, None] * error)[..., None, :]
        )
        outputs.append((q[:, i, :, :, None] * state).sum(-2))
    return torch.stack(outputs, 1), state


def section_op_parity(report, ops, device):
    import torch

    section = {"status": "running", "cases": []}
    report["sections"]["op_parity"] = section
    for label, b, t, h, dk, dv in GEOMETRIES:
        for dtype in DTYPES:
            entry = {"geometry": label, "dtype": dtype, "status": "running"}
            section["cases"].append(entry)
            try:
                xs = _inputs(torch, dtype, b, t, h, dk, dv, device)
                state = torch.randn(b, h, dk, dv, device=device) * 0.03
                d_out, d_fin, d_grads = _op_grads(torch, ops, xs, False, state)
                f_out, f_fin, f_grads = _op_grads(torch, ops, xs, True, state)
                pairs = [(f_out, d_out), (f_fin, d_fin), *zip(f_grads, d_grads)]
                rel = [_rel_l2(a, r) for a, r in pairs]
                bound = OP_REL_L2[dtype]
                entry["fused_vs_default_rel_l2"] = rel
                entry["bound"] = bound
                if max(rel) > bound:
                    raise ValueError(f"op parity {max(rel):.3e} > {bound:.3e}")
                entry["status"] = "passed"
                print(
                    f"D68_PARITY {label} {dtype} max_rel={max(rel):.2e} "
                    f"bound={bound:.0e}",
                    flush=True,
                )
            except Exception as exc:
                entry["status"] = "failed"
                entry["error"] = f"{type(exc).__name__}: {exc}"
    section["status"] = (
        "passed" if all(c["status"] == "passed" for c in section["cases"]) else "failed"
    )


def section_cpu_anchor(report, ops, device):
    import torch

    section = {"status": "running", "cases": []}
    report["sections"]["cpu_anchor"] = section
    for dtype in DTYPES:
        entry = {"dtype": dtype, "status": "running"}
        section["cases"].append(entry)
        try:
            xs = _inputs(torch, dtype, 1, 65, 2, 7, 9, device, seed=6807)
            state = torch.randn(1, 2, 7, 9, device=device) * 0.03
            refs = [x.detach().cpu().double().requires_grad_(True) for x in xs]
            sref = state.detach().cpu().double().requires_grad_(True)
            yr, sr = _recurrent_reference(torch, refs, True, sref)
            gy = torch.randn(1, 65, 2, 9, device=device) * 0.1
            gs = torch.randn(1, 2, 7, 9, device=device) * 0.1
            expected = torch.autograd.grad(
                (yr, sr), (*refs, sref), (gy.cpu().double(), gs.cpu().double())
            )
            atol, rtol = CPU_RECURRENCE_TOL[dtype]
            worst = 0.0
            for fused in (False, True):
                params = [x.detach().clone().requires_grad_(True) for x in xs]
                s0 = state.detach().clone().requires_grad_(True)
                y, s = ops.chunk_gated_delta_rule(
                    *params,
                    initial_state=s0,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                    use_fused_solve_scan=fused,
                )
                actual = torch.autograd.grad((y, s), (*params, s0), (gy, gs))
                for a, r in [(y, yr), (s, sr), *zip(actual, expected)]:
                    diff = (a.detach().cpu().double() - r).abs()
                    tol = atol + rtol * r.abs()
                    exceed = float((diff / tol).max())
                    worst = max(worst, exceed)
                    if exceed > 1.0:
                        raise ValueError(
                            f"fused={fused} CPU anchor exceeded by {exceed:.3f}x"
                        )
            entry["worst_tolerance_fraction"] = worst
            entry["tol"] = {"atol": atol, "rtol": rtol}
            entry["status"] = "passed"
            print(
                f"D68_CPU_ANCHOR {dtype} worst_tol_fraction={worst:.3f}",
                flush=True,
            )
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
    section["status"] = (
        "passed" if all(c["status"] == "passed" for c in section["cases"]) else "failed"
    )


def section_op_timing(report, ops, device):
    import torch

    section = {"status": "running", "cases": []}
    report["sections"]["op_timing"] = section
    for label, b, t, h, dk, dv in GEOMETRIES:
        for dtype in DTYPES:
            entry = {"geometry": label, "dtype": dtype, "status": "running"}
            section["cases"].append(entry)
            try:
                xs = _inputs(torch, dtype, b, t, h, dk, dv, device)
                for fused in (False, True):
                    for _ in range(OP_WARMUP):
                        _op_grads(torch, ops, xs, fused)
                torch.cuda.synchronize()
                samples = {False: [], True: []}
                for _ in range(OP_ROUNDS):
                    for fused in (False, True):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        _op_grads(torch, ops, xs, fused)
                        torch.cuda.synchronize()
                        samples[fused].append((time.perf_counter() - start) * 1000.0)
                entry["default"] = summarize(samples[False])
                entry["fused"] = summarize(samples[True])
                entry["speedup"] = speedup(
                    entry["default"]["median_ms"], entry["fused"]["median_ms"]
                )
                entry["status"] = "passed"
                print(
                    f"D68_OP {label} {dtype} "
                    f"default={entry['default']['median_ms']:.3f}ms "
                    f"fused={entry['fused']['median_ms']:.3f}ms "
                    f"speedup={entry['speedup']:.2f}x",
                    flush=True,
                )
            except Exception as exc:
                entry["status"] = "failed"
                entry["error"] = f"{type(exc).__name__}: {exc}"
    section["status"] = (
        "passed" if all(c["status"] == "passed" for c in section["cases"]) else "failed"
    )


def _decomposed_grads(torch, ops, xs, fuse_inverse, fuse_scan):
    """Replica of the op body with independent inverse/scan fusion switches.

    Measurement only: isolates the contribution of each fusion so the scan
    selectivity threshold is chosen from data, not assumption.
    """
    query, key, value, g, beta = [x.detach().clone().requires_grad_(True) for x in xs]
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
    attention = (q @ k.transpose(-1, -2)) * pair
    if fuse_inverse:
        inv = ops._FusedTrilInverse.apply(system)
        u = inv @ vb
        w = inv @ (kb * decay.exp().unsqueeze(-1))
    else:
        u = torch.linalg.solve_triangular(system, vb, upper=False, unitriangular=True)
        w = torch.linalg.solve_triangular(
            system, kb * decay.exp().unsqueeze(-1), upper=False, unitriangular=True
        )
    q = q * decay.exp().unsqueeze(-1)
    k = k * (decay[..., -1:] - decay).exp().unsqueeze(-1)
    chunk_decay = decay[..., -1].exp()[..., None, None]
    if fuse_scan:
        eye = torch.eye(dk, device=q.device, dtype=torch.float32)
        mm = chunk_decay * eye - k.transpose(-1, -2) @ w
        nn = k.transpose(-1, -2) @ u
        pp = q - attention @ w
        rr = attention @ u
        out, state = ops._FusedChunkScan.apply(mm, nn, pp, rr, None)
    else:
        state = torch.zeros((b, h, dk, dv), device=query.device, dtype=torch.float32)
        out = torch.zeros_like(u)
        for i in range(q.shape[2]):
            delta = u[:, :, i] - w[:, :, i] @ state
            out[:, :, i] = q[:, :, i] @ state + attention[:, :, i] @ delta
            state = state * chunk_decay[:, :, i] + k[:, :, i].transpose(-1, -2) @ delta
    loss = out.square().sum() + state.square().sum()
    grads = torch.autograd.grad(loss, (query, key, value, g, beta))
    return out, state, grads


def section_decomposition(report, ops, device):
    import torch

    section = {"status": "running", "cases": []}
    report["sections"]["decomposition"] = section
    variants = {
        "default": (False, False),
        "inverse_only": (True, False),
        "scan_only": (False, True),
        "both": (True, True),
    }
    for label, b, t, h, dk, dv in GEOMETRIES:
        for dtype in DTYPES:
            entry = {"geometry": label, "dtype": dtype, "status": "running"}
            section["cases"].append(entry)
            try:
                xs = _inputs(torch, dtype, b, t, h, dk, dv, device)
                base = _decomposed_grads(torch, ops, xs, False, False)
                for name, (fi, fs) in variants.items():
                    if name == "default":
                        continue
                    got = _decomposed_grads(torch, ops, xs, fi, fs)
                    rel = max(
                        _rel_l2(got[0], base[0]),
                        _rel_l2(got[1], base[1]),
                        *[_rel_l2(a, r) for a, r in zip(got[2], base[2])],
                    )
                    if rel > OP_REL_L2[dtype]:
                        raise ValueError(
                            f"{name} replica parity {rel:.3e} > {OP_REL_L2[dtype]:.3e}"
                        )
                    for _ in range(OP_WARMUP):
                        _decomposed_grads(torch, ops, xs, fi, fs)
                torch.cuda.synchronize()
                samples = {name: [] for name in variants}
                for _ in range(OP_ROUNDS):
                    for name, (fi, fs) in variants.items():
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        _decomposed_grads(torch, ops, xs, fi, fs)
                        torch.cuda.synchronize()
                        samples[name].append((time.perf_counter() - start) * 1000.0)
                for name in variants:
                    entry[name] = summarize(samples[name])
                    entry[f"{name}_speedup"] = speedup(
                        entry["default"]["median_ms"], entry[name]["median_ms"]
                    )
                entry["status"] = "passed"
                print(
                    f"D68_DECOMP {label} {dtype} "
                    + " ".join(
                        f"{name}={entry[name]['median_ms']:.3f}ms"
                        f"({entry[f'{name}_speedup']:.2f}x)"
                        for name in variants
                    ),
                    flush=True,
                )
            except Exception as exc:
                entry["status"] = "failed"
                entry["error"] = f"{type(exc).__name__}: {exc}"
    section["status"] = (
        "passed" if all(c["status"] == "passed" for c in section["cases"]) else "failed"
    )


def _install_fused_routing(model, ops):
    """D66 routing with the candidate fused op injected as the native call.

    The routed closure resolves routing._native_call at call time, so the
    injection must stay active until uninstall; the original is restored by
    _uninstall_routing via the record. The injection is GLOBAL: uninstall
    before installing again, or the next install captures the patched
    callable as its "original" and the restore chain leaks (D68 r5 trace).
    """
    original_native = routing._native_call

    def fused_native(query, key, value, g, beta, **kwargs):
        return ops.chunk_gated_delta_rule(
            query, key, value, g, beta, use_fused_solve_scan=True, **kwargs
        )

    routing._native_call = fused_native
    try:
        record = routing.install(model)
    except Exception:
        routing._native_call = original_native
        raise
    record["_d68_original_native"] = original_native
    print(
        f"D68_INJECT native_call {original_native.__name__} -> fused_native",
        flush=True,
    )
    return record


def _uninstall_routing(record):
    routing.uninstall(record)
    if "_d68_original_native" in record:
        restored = record.pop("_d68_original_native")
        print(
            f"D68_RESTORE native_call {routing._native_call.__name__} -> "
            f"{restored.__name__}",
            flush=True,
        )
        routing._native_call = restored
    else:
        print(
            f"D68_RESTORE no-key native_call={routing._native_call.__name__}",
            flush=True,
        )


def section_model_parity(report, ops, device):
    import torch

    section = {"status": "running", "dtypes": []}
    report["sections"]["model_parity"] = section
    original_seq = benchmark.SEQ
    # seq32 (C=1) verifies the flag's default-path selectivity against the
    # D66 routed parity; seq256 (C=4) qualifies the active fused path at model
    # level under the same pre-declared D66 bounds.
    for seq in (original_seq, 256):
        benchmark.SEQ = seq
        for dtype in DTYPES:
            entry = {"dtype": dtype, "seq": seq, "status": "running"}
            section["dtypes"].append(entry)
            record = None
            try:
                torch.manual_seed(2662)
                frozen = admission.build_model(MODEL, "float32", BACKEND, device)
                torch.manual_seed(2662)
                routed = admission.build_model(MODEL, "float32", BACKEND, device)
                record = _install_fused_routing(routed, ops)
                vocab = admission.registry_spec(MODEL)["config"]["vocab_size"]
                torch.manual_seed(2663)
                ids = torch.randint(
                    0, vocab, (benchmark.BATCH, benchmark.SEQ), device=device
                )
                compute = getattr(torch, dtype)
                outputs, grads = [], []
                for model in (frozen, routed):
                    model.zero_grad(set_to_none=True)
                    with torch.autocast(
                        "cuda", dtype=compute, enabled=(dtype != "float32")
                    ):
                        out = model(input_ids=ids, labels=ids, use_cache=False)
                    out.loss.backward()
                    outputs.append((out.loss.detach(), out.logits.detach()))
                    grads.append(
                        {
                            name: p.grad.detach()
                            for name, p in model.named_parameters()
                            if p.grad is not None
                        }
                    )
                loss_delta = float((outputs[0][0] - outputs[1][0]).abs())
                logits_rel = _rel_l2(outputs[1][1], outputs[0][1])
                grad_rel = {
                    name: _rel_l2(grads[1][name], grads[0][name]) for name in grads[0]
                }
                worst = sorted(grad_rel.items(), key=lambda kv: kv[1], reverse=True)[:5]
                if dtype == "float32":
                    loss_bound = MODEL_FP32_LOSS_ATOL
                    logits_bound = MODEL_FP32_LOGITS_REL_L2
                    grad_bound = MODEL_FP32_GRAD_REL_L2
                else:
                    loss_bound = MODEL_LOW_LOSS_ATOL
                    logits_bound = MODEL_LOW_GRAD_REL_L2
                    grad_bound = MODEL_LOW_GRAD_REL_L2
                entry.update(
                    {
                        "loss_delta": loss_delta,
                        "loss_bound": loss_bound,
                        "logits_rel_l2": logits_rel,
                        "logits_bound": logits_bound,
                        "max_grad_rel_l2": max(grad_rel.values()),
                        "grad_bound": grad_bound,
                        "worst_grads": [{"name": n, "rel_l2": v} for n, v in worst],
                        "native_calls": record["native_calls"],
                        "fallbacks": record["fallbacks"],
                    }
                )
                if dtype == "float32":
                    entry["cpu_anchor"] = _cpu_grad_anchor(
                        torch, frozen, grads, ids, device
                    )
                if loss_delta > loss_bound:
                    raise ValueError(f"loss delta {loss_delta:.3e} > {loss_bound:.3e}")
                if logits_rel > logits_bound:
                    raise ValueError(
                        f"logits rel_l2 {logits_rel:.3e} > {logits_bound:.3e}"
                    )
                if max(grad_rel.values()) > grad_bound:
                    raise ValueError(
                        f"grad rel_l2 {max(grad_rel.values()):.3e} > "
                        f"{grad_bound:.3e}"
                    )
                if record["native_calls"] == 0 or record["fallbacks"]:
                    raise ValueError(f"unexpected routing record {record}")
                entry["status"] = "passed"
                print(
                    f"D68_MODEL_PARITY seq{seq} {dtype} "
                    f"loss_delta={loss_delta:.2e} "
                    f"logits_rel={logits_rel:.2e} "
                    f"max_grad_rel={max(grad_rel.values()):.2e} "
                    f"native_calls={record['native_calls']}",
                    flush=True,
                )
                del frozen, routed
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as exc:
                entry["status"] = "failed"
                entry["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                if record is not None:
                    _uninstall_routing(record)
    benchmark.SEQ = original_seq
    section["status"] = (
        "passed"
        if all(d["status"] == "passed" for d in section["dtypes"])
        else "failed"
    )


def _cpu_grad_anchor(torch, frozen, grads, ids, device) -> dict:
    """Independent FP64 CPU anchor: exact frozen weights, same forward/backward."""
    cpu_model = admission.build_model(MODEL, "float32", BACKEND, "cpu").double()
    cpu_model.load_state_dict(
        {k: v.detach().cpu().double() for k, v in frozen.state_dict().items()}
    )
    cpu_model.zero_grad(set_to_none=True)
    cpu_ids = ids.detach().cpu()
    out = cpu_model(input_ids=cpu_ids, labels=cpu_ids, use_cache=False)
    out.loss.backward()
    cpu_grads = {
        name: p.grad.detach()
        for name, p in cpu_model.named_parameters()
        if p.grad is not None
    }
    frozen_rel, fused_rel = {}, {}
    for name in cpu_grads:
        reference = cpu_grads[name]
        frozen_rel[name] = _rel_l2(grads[0][name].cpu(), reference)
        fused_rel[name] = _rel_l2(grads[1][name].cpu(), reference)
    worst = sorted(
        (
            {"name": n, "frozen": frozen_rel[n], "fused": fused_rel[n]}
            for n in cpu_grads
        ),
        key=lambda e: max(e["frozen"], e["fused"]),
        reverse=True,
    )[:5]
    anchor = {
        "bound": CPU_ANCHOR_GRAD_REL_L2,
        "frozen_max_rel_l2": max(frozen_rel.values()),
        "fused_max_rel_l2": max(fused_rel.values()),
        "worst": worst,
    }
    if max(anchor["frozen_max_rel_l2"], anchor["fused_max_rel_l2"]) > (
        CPU_ANCHOR_GRAD_REL_L2
    ):
        raise ValueError(
            "CPU/FP64 gradient anchor exceeded: "
            f"frozen={anchor['frozen_max_rel_l2']:.3e} "
            f"fused={anchor['fused_max_rel_l2']:.3e} > {CPU_ANCHOR_GRAD_REL_L2:.3e}"
        )
    del cpu_model
    gc.collect()
    return anchor


def section_model_timing(report, ops, device):
    import torch

    section = {"status": "running", "variants": []}
    report["sections"]["model_timing"] = section
    original_build = admission.build_model
    original_seq = benchmark.SEQ
    # Count fused-kernel invocations per variant so the report proves which
    # path each timing actually exercised (diag2, D68).
    scan_calls = {"n": 0}
    original_scan_apply = ops._FusedChunkScan.apply

    def counting_scan(*args, **kwargs):
        scan_calls["n"] += 1
        return original_scan_apply(*args, **kwargs)

    ops._FusedChunkScan.apply = staticmethod(counting_scan)
    # SEQ=32 is the D64/D66 tiny geometry (one chunk: the fused flag keeps the
    # default path there); SEQ=256 spans four chunks and exercises the fused
    # scan at model level. benchmark_case reads the module global at call time.
    for seq in (original_seq, 256):
        benchmark.SEQ = seq
        for variant in ("frozen", "routed", "routed_fused"):
            records = []
            variant_native = routing._native_call

            def variant_build(name, dtype, backend, dev):
                model = original_build(name, dtype, backend, dev)
                if name == MODEL:
                    if variant == "routed":
                        records.append(routing.install(model))
                    elif variant == "routed_fused":
                        records.append(_install_fused_routing(model, ops))
                return model

            admission.build_model = variant_build
            try:
                for dtype in DTYPES:
                    scan_calls["n"] = 0
                    bench_report = benchmark.new_report()
                    benchmark.benchmark_case(
                        bench_report, MODEL, dtype, BACKEND, device
                    )
                    case = bench_report["cases"][0]
                    entry = {
                        "variant": variant,
                        "dtype": dtype,
                        "seq": seq,
                        "fused_scan_calls": scan_calls["n"],
                        "native_call_after": routing._native_call.__name__,
                        **case,
                    }
                    section["variants"].append(entry)
                    if case["status"] == "passed":
                        print(
                            f"D68_MODEL_TIMING seq{seq} {variant} {dtype} "
                            f"step_ms={case['step_median_ms']:.3f} "
                            f"tokens_s={case['tokens_per_second']:.1f} "
                            f"first_ms={case['first_step_ms']:.1f} "
                            f"scan_calls={scan_calls['n']}",
                            flush=True,
                        )
                    else:
                        print(
                            f"D68_MODEL_TIMING seq{seq} {variant} {dtype} FAILED "
                            f"{case.get('error', '')}",
                            flush=True,
                        )
                    # Uninstall per dtype: the _native_call injection is
                    # global, so a second install before the first uninstall
                    # would capture the patched callable (D68 r5 root cause).
                    for record in records:
                        _uninstall_routing(record)
                    records.clear()
            finally:
                admission.build_model = original_build
                for record in records:
                    _uninstall_routing(record)
                routing._native_call = variant_native
    benchmark.SEQ = original_seq
    ops._FusedChunkScan.apply = original_scan_apply
    for seq in (original_seq, 256):
        for dtype in DTYPES:
            by_variant = {
                v["variant"]: v
                for v in section["variants"]
                if v["dtype"] == dtype and v["seq"] == seq and v["status"] == "passed"
            }
            if "frozen" in by_variant:
                for name in ("routed", "routed_fused"):
                    if name in by_variant:
                        by_variant[name]["speedup_vs_frozen"] = speedup(
                            by_variant["frozen"]["step_median_ms"],
                            by_variant[name]["step_median_ms"],
                        )
            if "routed" in by_variant and "routed_fused" in by_variant:
                by_variant["routed_fused"]["speedup_vs_routed"] = speedup(
                    by_variant["routed"]["step_median_ms"],
                    by_variant["routed_fused"]["step_median_ms"],
                )
    section["status"] = (
        "passed"
        if all(v["status"] == "passed" for v in section["variants"])
        else "failed"
    )


def section_refusals(report, ops, installed, device):
    import torch

    section = {"status": "running", "checks": []}
    report["sections"]["refusals"] = section

    def check(name, fn):
        entry = {"name": name, "status": "running"}
        section["checks"].append(entry)
        try:
            fn()
            entry["status"] = "passed"
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"

    xs = _inputs(torch, "float32", 1, 7, 2, 7, 9, device)

    def non_bool():
        for bad in (1, "yes", None):
            try:
                ops.chunk_gated_delta_rule(*xs, use_fused_solve_scan=bad)
            except TypeError:
                continue
            raise ValueError(f"non-bool flag {bad!r} was accepted")

    def state_cap():
        big = _inputs(torch, "float32", 33, 512, 16, 128, 128, device)
        try:
            ops.chunk_gated_delta_rule(*big, use_fused_solve_scan=True)
        except ValueError:
            return
        raise ValueError("oversized fused state buffer was accepted")

    def default_invariance():
        state = torch.randn(1, 2, 7, 9, device=device) * 0.03
        a = _op_grads(torch, ops, xs, False, state)
        b = _op_grads(torch, installed, xs, None, state)
        for ta, tb in zip(a, b):
            if isinstance(ta, tuple):
                for x, y in zip(ta, tb):
                    if not torch.equal(x, y):
                        raise ValueError("candidate default path diverges (grads)")
            elif not torch.equal(ta, tb):
                raise ValueError("candidate default path diverges")

    check("non_bool_flag_refused", non_bool)
    check("state_cap_refused", state_cap)
    check("default_path_bitwise_invariant", default_invariance)
    section["status"] = (
        "passed"
        if all(c["status"] == "passed" for c in section["checks"])
        else "failed"
    )
    print(
        f"D68_REFUSALS {section['status']} "
        f"({sum(c['status'] == 'passed' for c in section['checks'])} checks)",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import torch
    import triton
    from torch.nn import bc250_gated_delta as installed

    candidate_path = os.environ.get("BC250_D68_CANDIDATE")
    if not candidate_path:
        raise SystemExit("BC250_D68_CANDIDATE must point at the candidate module")
    ops = load_candidate(candidate_path)

    report = new_report()
    report["versions"] = {
        "torch": torch.__version__,
        "triton": triton.__version__,
        "transformers": __import__("transformers").__version__,
        "python": sys.version.split()[0],
    }
    report["modules"] = {
        "candidate": {
            "path": candidate_path,
            "sha256": hashlib.sha256(Path(candidate_path).read_bytes()).hexdigest(),
        },
        "installed": {
            "path": installed.__file__,
            "sha256": hashlib.sha256(Path(installed.__file__).read_bytes()).hexdigest(),
        },
    }
    props = torch.cuda.get_device_properties(args.device)
    report["device"] = {"name": props.name, "gcnArchName": props.gcnArchName}
    print(
        f"D68 candidate={report['modules']['candidate']['sha256'][:12]} "
        f"installed={report['modules']['installed']['sha256'][:12]} "
        f"torch={torch.__version__} triton={triton.__version__}",
        flush=True,
    )
    section_op_parity(report, ops, args.device)
    section_cpu_anchor(report, ops, args.device)
    section_op_timing(report, ops, args.device)
    section_decomposition(report, ops, args.device)
    section_model_parity(report, ops, args.device)
    section_model_timing(report, ops, args.device)
    section_refusals(report, ops, installed, args.device)
    failed = [
        name
        for name, section in report["sections"].items()
        if section["status"] != "passed"
    ]
    report["status"] = "passed" if not failed else "failed"
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(f"QUALIFY_D68 {report['status']} failed_sections={failed}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
