#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""D69 route-port qualification in the app's Transformers 5.22.1 environment.

Runs inside the qualified worker with PYTHONPATH pointing at the application
site-packages (the guardian/bounded launcher provides this). Compares the
5.22.1 reference chunk implementation against the installed native op and the
staged D68 fused candidate, qualifies a tiny owned Qwen3.5 text model under the
module-scope route (both targets), and exercises the route's refusals/fallbacks.

Pre-declared bounds (D69, set before any run; no post-hoc relaxation):
- op rel-L2 vs the 5.22.1 reference: FP32 1e-5, BF16/FP16 1e-2 (D66 op bounds);
- op CPU/FP64 recurrence anchor: the A06 recurrent tolerances per dtype;
- tiny-model parity: the D66 model bounds + CPU/FP64 gradient anchor 5e-3.

Owned-only: random inputs and a randomly initialized tiny model; no user data,
no application file modified, nothing installed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import route_qwen35_gated_delta_d69 as routing  # noqa: E402

FIXTURE = "qualify_route_d69"
DTYPES = ("float32", "bfloat16", "float16")
# label, batch, seq, heads, dk, dv (the last two match the 2B linear geometry)
GEOMETRIES = (
    ("tiny", 2, 32, 2, 16, 16),
    ("long256", 2, 256, 2, 16, 16),
    ("2b-heads-256", 1, 256, 16, 128, 128),
    ("2b-heads-1024", 1, 1024, 16, 128, 128),
)
OP_REL_L2 = {"float32": 1e-5, "bfloat16": 1e-2, "float16": 1e-2}
CPU_RECURRENCE_TOL = {
    "float32": (2e-5, 2e-4),
    "float16": (0.001, 0.008),
    "bfloat16": (0.006, 0.04),
}
MODEL_FP32_LOSS_ATOL = 1e-4
MODEL_FP32_LOGITS_REL_L2 = 1e-4
MODEL_FP32_GRAD_REL_L2 = 2e-3
MODEL_LOW_LOSS_ATOL = 0.015
MODEL_LOW_GRAD_REL_L2 = 0.04
CPU_ANCHOR_GRAD_REL_L2 = 5e-3
MODEL_SEQ = 64
MODEL_BATCH = 2


# ---------------------------------------------------------------------------
# Pure logic (workstation-testable).
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
            "model_low_loss_atol": MODEL_LOW_LOSS_ATOL,
            "model_low_grad_rel_l2": MODEL_LOW_GRAD_REL_L2,
            "cpu_anchor_grad_rel_l2": CPU_ANCHOR_GRAD_REL_L2,
        },
        "geometries": [
            {"label": label, "batch": b, "seq": t, "heads": h, "dk": dk, "dv": dv}
            for label, b, t, h, dk, dv in GEOMETRIES
        ],
        "versions": {},
        "modules": {},
        "sections": {},
    }


def tiny_config_overrides() -> dict:
    return {
        "vocab_size": 256,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "max_position_embeddings": 512,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4,
        "tie_word_embeddings": True,
        "attention_dropout": 0.0,
        "full_attention_interval": 4,
    }


# ---------------------------------------------------------------------------
# Torch-dependent sections.
# ---------------------------------------------------------------------------
def _rel_l2(actual, reference) -> float:
    a = actual.detach().double()
    b = reference.detach().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def _inputs(torch, dtype, batch, seq, heads, dk, dv, device, seed=6901):
    compute = getattr(torch, dtype)
    torch.manual_seed(seed)
    q = torch.randn(batch, seq, heads, dk, device=device, dtype=compute) * 0.15
    k = torch.randn(batch, seq, heads, dk, device=device, dtype=compute) * 0.15
    v = torch.randn(batch, seq, heads, dv, device=device, dtype=compute) * 0.2
    g = -(torch.rand(batch, seq, heads, device=device) * 0.15 + 0.01)
    beta = torch.rand(batch, seq, heads, device=device) * 0.8 + 0.1
    return q, k, v, g, beta


def _op_grads(torch, fn, xs, state=None, fused=False):
    params = [x.detach().clone().requires_grad_(True) for x in xs]
    s0 = None if state is None else state.detach().clone().requires_grad_(True)
    kwargs = {"use_fused_solve_scan": True} if fused else {}
    out, final = fn(
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


def section_op_parity(report, ctx, device):
    import torch

    section = {"status": "running", "cases": []}
    report["sections"]["op_parity"] = section
    reference, installed, candidate = (
        ctx["reference"],
        ctx["installed"],
        ctx["candidate"],
    )
    for label, b, t, h, dk, dv in GEOMETRIES:
        for dtype in DTYPES:
            entry = {"geometry": label, "dtype": dtype, "status": "running"}
            section["cases"].append(entry)
            try:
                xs = _inputs(torch, dtype, b, t, h, dk, dv, device)
                state = torch.randn(b, h, dk, dv, device=device) * 0.03
                ref = _op_grads(torch, reference, xs, state)
                ins = _op_grads(torch, installed.chunk_gated_delta_rule, xs, state)
                fus = _op_grads(
                    torch, candidate.chunk_gated_delta_rule, xs, state, fused=True
                )
                bound = OP_REL_L2[dtype]
                rel_ins = [
                    _rel_l2(a, r)
                    for a, r in [
                        (ins[0], ref[0]),
                        (ins[1], ref[1]),
                        *zip(ins[2], ref[2]),
                    ]
                ]
                rel_fus = [
                    _rel_l2(a, r)
                    for a, r in [
                        (fus[0], ref[0]),
                        (fus[1], ref[1]),
                        *zip(fus[2], ref[2]),
                    ]
                ]
                entry["installed_vs_reference_rel_l2"] = rel_ins
                entry["fused_vs_reference_rel_l2"] = rel_fus
                entry["bound"] = bound
                if max(rel_ins) > bound or max(rel_fus) > bound:
                    raise ValueError(
                        f"op parity installed={max(rel_ins):.3e} "
                        f"fused={max(rel_fus):.3e} > {bound:.3e}"
                    )
                entry["status"] = "passed"
                print(
                    f"D69_OP_PARITY {label} {dtype} "
                    f"installed={max(rel_ins):.2e} fused={max(rel_fus):.2e} "
                    f"bound={bound:.0e}",
                    flush=True,
                )
            except Exception as exc:
                entry["status"] = "failed"
                entry["error"] = f"{type(exc).__name__}: {exc}"
    section["status"] = (
        "passed" if all(c["status"] == "passed" for c in section["cases"]) else "failed"
    )


def section_cpu_anchor(report, ctx, device):
    import torch

    section = {"status": "running", "cases": []}
    report["sections"]["cpu_anchor"] = section
    reference, installed, candidate = (
        ctx["reference"],
        ctx["installed"],
        ctx["candidate"],
    )
    for dtype in DTYPES:
        entry = {"dtype": dtype, "status": "running"}
        section["cases"].append(entry)
        try:
            xs = _inputs(torch, dtype, 1, 65, 2, 7, 9, device, seed=6907)
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
            variants = (
                ("reference", reference, False),
                ("installed", installed.chunk_gated_delta_rule, False),
                ("fused", candidate.chunk_gated_delta_rule, True),
            )
            for name, fn, fused in variants:
                params = [x.detach().clone().requires_grad_(True) for x in xs]
                s0 = state.detach().clone().requires_grad_(True)
                kwargs = {"use_fused_solve_scan": True} if fused else {}
                y, s = fn(
                    *params,
                    initial_state=s0,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                    **kwargs,
                )
                actual = torch.autograd.grad((y, s), (*params, s0), (gy, gs))
                for a, r in [(y, yr), (s, sr), *zip(actual, expected)]:
                    diff = (a.detach().cpu().double() - r).abs()
                    exceed = float((diff / (atol + rtol * r.abs())).max())
                    worst = max(worst, exceed)
                    if exceed > 1.0:
                        raise ValueError(f"{name} CPU anchor exceeded by {exceed:.3f}x")
            entry["worst_tolerance_fraction"] = worst
            entry["status"] = "passed"
            print(f"D69_CPU_ANCHOR {dtype} worst_tol_fraction={worst:.3f}", flush=True)
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
    section["status"] = (
        "passed" if all(c["status"] == "passed" for c in section["cases"]) else "failed"
    )


def _build_tiny(torch, transformers, device, dtype="float32"):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    config = Qwen3_5TextConfig(**tiny_config_overrides())
    model = Qwen3_5ForCausalLM(config)
    return model.to(getattr(torch, dtype)).to(device).train()


def _model_forward_backward(torch, model, ids, compute_dtype):
    model.zero_grad(set_to_none=True)
    with torch.autocast(
        "cuda", dtype=compute_dtype, enabled=(compute_dtype != torch.float32)
    ):
        out = model(input_ids=ids, labels=ids, use_cache=False)
    out.loss.backward()
    grads = {
        name: p.grad.detach()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    return out.loss.detach(), out.logits.detach(), grads


def section_model_parity(report, ctx, device):
    import torch
    import transformers

    section = {"status": "running", "cases": []}
    report["sections"]["model_parity"] = section
    candidate_path = ctx["candidate_path"]
    for dtype in DTYPES:
        entry = {"dtype": dtype, "status": "running"}
        section["cases"].append(entry)
        records = []
        try:
            torch.manual_seed(2692)
            frozen = _build_tiny(torch, transformers, device)
            variants = {}
            # reference (no route)
            torch.manual_seed(2693)
            ids = torch.randint(0, 256, (MODEL_BATCH, MODEL_SEQ), device=device)
            compute = getattr(torch, dtype)
            variants["reference"] = _model_forward_backward(torch, frozen, ids, compute)
            # installed and fused routes, installed/uninstalled per variant
            for target in ("installed", "fused"):
                torch.manual_seed(2692)
                routed_model = _build_tiny(torch, transformers, device)
                record = routing.install(
                    target=target,
                    candidate_path=candidate_path if target == "fused" else None,
                )
                records.append(record)
                variants[target] = _model_forward_backward(
                    torch, routed_model, ids, compute
                )
                routing.uninstall(record)
                records.pop()
                if record["native_calls"] == 0 or record["fallbacks"]:
                    raise ValueError(f"{target} routing record {record}")
                entry[f"{target}_native_calls"] = record["native_calls"]
                del routed_model
                gc.collect()
                torch.cuda.empty_cache()
            ref_loss, ref_logits, ref_grads = variants["reference"]
            if dtype == "float32":
                loss_bound = MODEL_FP32_LOSS_ATOL
                logits_bound = MODEL_FP32_LOGITS_REL_L2
                grad_bound = MODEL_FP32_GRAD_REL_L2
            else:
                loss_bound = MODEL_LOW_LOSS_ATOL
                logits_bound = MODEL_LOW_GRAD_REL_L2
                grad_bound = MODEL_LOW_GRAD_REL_L2
            for target in ("installed", "fused"):
                loss, logits, grads = variants[target]
                loss_delta = float((loss - ref_loss).abs())
                logits_rel = _rel_l2(logits, ref_logits)
                grad_rel = {
                    name: _rel_l2(grads[name], ref_grads[name]) for name in ref_grads
                }
                worst = sorted(grad_rel.items(), key=lambda kv: kv[1], reverse=True)[:5]
                entry[target] = {
                    "loss_delta": loss_delta,
                    "logits_rel_l2": logits_rel,
                    "max_grad_rel_l2": max(grad_rel.values()),
                    "worst_grads": [{"name": n, "rel_l2": v} for n, v in worst],
                }
                if loss_delta > loss_bound:
                    raise ValueError(
                        f"{target} loss delta {loss_delta:.3e} > {loss_bound:.3e}"
                    )
                if logits_rel > logits_bound:
                    raise ValueError(
                        f"{target} logits {logits_rel:.3e} > {logits_bound:.3e}"
                    )
                if max(grad_rel.values()) > grad_bound:
                    raise ValueError(
                        f"{target} grad {max(grad_rel.values()):.3e} > {grad_bound:.3e}"
                    )
            if dtype == "float32":
                entry["cpu_anchor"] = _model_cpu_anchor(
                    torch, transformers, frozen, variants, ids
                )
            entry["status"] = "passed"
            print(
                f"D69_MODEL_PARITY {dtype} "
                + " ".join(
                    f"{t}_grad={entry[t]['max_grad_rel_l2']:.2e}"
                    for t in ("installed", "fused")
                ),
                flush=True,
            )
            del frozen
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            for record in records:
                routing.uninstall(record)
    section["status"] = (
        "passed" if all(c["status"] == "passed" for c in section["cases"]) else "failed"
    )


def _model_cpu_anchor(torch, transformers, frozen, variants, ids) -> dict:
    cpu_model = _build_tiny(torch, transformers, "cpu").double()
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
    result = {"bound": CPU_ANCHOR_GRAD_REL_L2}
    for target in ("reference", "installed", "fused"):
        grads = variants[target][2]
        rel = {name: _rel_l2(grads[name].cpu(), cpu_grads[name]) for name in cpu_grads}
        result[f"{target}_max_rel_l2"] = max(rel.values())
        if max(rel.values()) > CPU_ANCHOR_GRAD_REL_L2:
            raise ValueError(
                f"model CPU anchor {target} {max(rel.values()):.3e} > "
                f"{CPU_ANCHOR_GRAD_REL_L2:.3e}"
            )
    del cpu_model
    gc.collect()
    return result


def section_refusals(report, ctx, device):
    import torch
    import transformers

    section = {"status": "running", "checks": []}
    report["sections"]["refusals"] = section
    candidate_path = ctx["candidate_path"]
    installed_path = ctx["installed_path"]
    reference = ctx["reference"]
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling

    def check(name, fn):
        entry = {"name": name, "status": "running"}
        section["checks"].append(entry)
        try:
            fn()
            entry["status"] = "passed"
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"

    def expect_value_error(fn):
        try:
            fn()
        except ValueError:
            return
        raise ValueError("expected ValueError")

    xs = _inputs(torch, "float32", 1, 65, 2, 7, 9, device)

    def kill_switch():
        expect_value_error(
            lambda: routing.install(
                target="installed", env={"BC250_NATIVE_GATED_DELTA": "0"}
            )
        )

    def version_gate():
        saved = transformers.__version__
        transformers.__version__ = "5.2.0"
        try:
            expect_value_error(lambda: routing.install(target="installed"))
        finally:
            transformers.__version__ = saved

    def bad_target():
        expect_value_error(lambda: routing.install(target="bogus"))

    def fused_requires_candidate():
        expect_value_error(lambda: routing.install(target="fused"))
        expect_value_error(
            lambda: routing.install(target="fused", candidate_path=installed_path)
        )

    def uninstall_identity():
        record = routing.install(target="installed")
        routing.uninstall(record)
        expect_value_error(lambda: routing.uninstall(record))
        record = routing.install(target="installed")
        foreign = lambda *a, **k: None  # noqa: E731
        modeling.torch_chunk_gated_delta_rule = foreign
        try:
            expect_value_error(lambda: routing.uninstall(record))
        finally:
            modeling.torch_chunk_gated_delta_rule = record["original"]

    def fallbacks_counted():
        record = routing.install(target="fused", candidate_path=candidate_path)
        try:
            state = torch.randn(1, 2, 7, 9, device=device) * 0.03
            base = reference(
                *xs,
                initial_state=None,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            # cu_seqlens=None is the ordinary unpacked contract -> native.
            got = modeling.torch_chunk_gated_delta_rule(
                *xs,
                initial_state=None,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=None,
                num_items_in_batch=4,
            )
            if record["native_calls"] != 1 or record["fallbacks"]:
                raise ValueError(f"cu_seqlens=None should route: {record}")
            if "num_items_in_batch" not in record.get("ignored_control_kwargs", set()):
                raise ValueError("control kwarg not recorded as ignored")
            # Native output is a different FP32 rounding, not bitwise equal.
            if _rel_l2(got[0], base[0]) > OP_REL_L2["float32"]:
                raise ValueError("unpacked routed output diverges from reference")
            # Packed sequences (non-None cu_seqlens) -> counted fallback. The
            # 5.22.1 torch reference itself refuses packed sequences without
            # flash-linear-attention; the wrapper must forward and let it
            # raise, not silently coerce.
            packed = torch.tensor([0, 32, 65], device=device, dtype=torch.int32)
            try:
                modeling.torch_chunk_gated_delta_rule(
                    *xs,
                    initial_state=None,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                    cu_seqlens=packed,
                )
            except ValueError:
                pass
            else:
                raise ValueError("packed reference call unexpectedly succeeded")
            if record["fallbacks"].get("cu-seqlens", 0) != 1:
                raise ValueError(
                    f"cu-seqlens fallback not counted: {record['fallbacks']}"
                )
            modeling.torch_chunk_gated_delta_rule(
                *xs,
                initial_state=state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            if record["fallbacks"].get("initial-state", 0) != 1:
                raise ValueError("initial-state fallback not counted")
            big = _inputs(torch, "float32", 1, 65, 2, 256, 9, device)
            modeling.torch_chunk_gated_delta_rule(
                *big, output_final_state=True, use_qk_l2norm_in_kernel=True
            )
            if record["fallbacks"].get("geometry", 0) != 1:
                raise ValueError("geometry fallback not counted")
            ints = tuple(x.to(torch.int32) for x in xs[:3]) + xs[3:]
            modeling.torch_chunk_gated_delta_rule(
                *ints, output_final_state=True, use_qk_l2norm_in_kernel=True
            )
            if record["fallbacks"].get("dtype", 0) != 1:
                raise ValueError("dtype fallback not counted")
            if record["native_calls"] != 1:
                raise ValueError(f"unexpected native calls: {record['native_calls']}")
        finally:
            routing.uninstall(record)

    def native_dispatch():
        record = routing.install(target="fused", candidate_path=candidate_path)
        try:
            out, final = modeling.torch_chunk_gated_delta_rule(
                *xs, output_final_state=True, use_qk_l2norm_in_kernel=True
            )
            if record["native_calls"] != 1 or record["fallbacks"]:
                raise ValueError(f"dispatch record {record}")
            direct = ctx["candidate"].chunk_gated_delta_rule(
                *xs,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_fused_solve_scan=True,
            )
            if not torch.equal(out, direct[0]):
                raise ValueError("routed output differs from direct fused call")
        finally:
            routing.uninstall(record)

    check("kill_switch_refused", kill_switch)
    check("version_gate_refused", version_gate)
    check("bad_target_refused", bad_target)
    check("fused_requires_d68_candidate", fused_requires_candidate)
    check("uninstall_identity", uninstall_identity)
    check("fallbacks_counted", fallbacks_counted)
    check("native_dispatch_matches_direct", native_dispatch)
    section["status"] = (
        "passed"
        if all(c["status"] == "passed" for c in section["checks"])
        else "failed"
    )
    print(
        f"D69_REFUSALS {section['status']} "
        f"({sum(c['status'] == 'passed' for c in section['checks'])} checks)",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--candidate",
        default=str(Path(__file__).resolve().parent / "bc250_gated_delta.py"),
    )
    args = parser.parse_args()

    import importlib.util

    import torch
    import triton
    import transformers
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    import torch.nn.bc250_gated_delta as installed

    spec_candidate = importlib.util.spec_from_file_location(
        "gated_delta_d69_candidate", args.candidate
    )
    candidate = importlib.util.module_from_spec(spec_candidate)
    spec_candidate.loader.exec_module(candidate)

    report = new_report()
    report["versions"] = {
        "torch": torch.__version__,
        "triton": triton.__version__,
        "transformers": transformers.__version__,
        "python": sys.version.split()[0],
    }
    report["modules"] = {
        "modeling_sha256": hashlib.sha256(
            Path(modeling.__file__).read_bytes()
        ).hexdigest(),
        "installed_native": {
            "path": installed.__file__,
            "sha256": hashlib.sha256(Path(installed.__file__).read_bytes()).hexdigest(),
        },
        "candidate": {
            "path": args.candidate,
            "sha256": hashlib.sha256(Path(args.candidate).read_bytes()).hexdigest(),
        },
        "route_helper_sha256": routing.sha256_file(Path(routing.__file__)),
    }
    props = torch.cuda.get_device_properties(args.device)
    report["device"] = {"name": props.name, "gcnArchName": props.gcnArchName}
    print(
        f"D69 transformers={transformers.__version__} torch={torch.__version__} "
        f"candidate={report['modules']['candidate']['sha256'][:12]}",
        flush=True,
    )
    ctx = {
        "reference": modeling.torch_chunk_gated_delta_rule,
        "installed": installed,
        "candidate": candidate,
        "candidate_path": args.candidate,
        "installed_path": installed.__file__,
    }
    section_op_parity(report, ctx, args.device)
    section_cpu_anchor(report, ctx, args.device)
    section_model_parity(report, ctx, args.device)
    section_refusals(report, ctx, args.device)
    failed = [
        name
        for name, section in report["sections"].items()
        if section["status"] != "passed"
    ]
    report["status"] = "passed" if not failed else "failed"
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(f"QUALIFY_D69 {report['status']} failed_sections={failed}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
