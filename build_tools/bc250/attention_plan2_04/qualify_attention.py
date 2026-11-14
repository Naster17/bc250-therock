#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PLAN2-04 attention GPU qualification (inside the guardian).

Part 1: small/mid parity grid, refusal surface, checkpoint/Transformers/autocast
integration, empty dims, informational CUDA-graph probe. Part 2: long-sequence
parity grid (L 1024-4096). Qualifies the INSTALLED opt-in tiled SDPA against GPU
math SDPA and an independent CPU-FP64 reference within the pre-declared bounds
(tmp/attention-plan2-04/BOUNDS.md). No operator/dispatch/default/source change.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402
from torch.nn.attention import _gfx1013 as tiled_mod  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

from attention_plan2_04 import fixtures  # noqa: E402

TILED = tiled_mod.scaled_dot_product_attention
REPORT = {"checks": [], "environment": {}, "part": None}
STARTED = time.monotonic()


def check(name, passed, **detail):
    REPORT["checks"].append({"name": name, "passed": bool(passed), **detail})
    if not passed:
        print(f"ATTN_FAIL {name} {detail}", flush=True)
    return passed


def save(output):
    REPORT["seconds"] = time.monotonic() - STARTED
    REPORT["summary"] = {
        "total": len(REPORT["checks"]),
        "passed": sum(c["passed"] for c in REPORT["checks"]),
    }
    REPORT["summary"]["all_passed"] = REPORT["summary"]["passed"] == len(
        REPORT["checks"]
    )
    output.write_text(json.dumps(REPORT, indent=2) + "\n")


def rel_l2(actual, expected):
    a, b = actual.detach().double(), expected.detach().double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def cpu_reference(q, k, v, mask, causal, scale, gqa):
    """Independent CPU-double attention; fully masked rows are zero (SDPA math)."""
    if gqa:
        repeat = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(repeat, 1)
        v = v.repeat_interleave(repeat, 1)
    scores = (q @ k.transpose(-1, -2)) * scale
    if causal:
        allowed = torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool).tril()
        scores = scores.masked_fill(~allowed, -float("inf"))
    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, -float("inf"))
        else:
            scores = scores + mask
    empty = (scores == -float("inf")).all(-1, keepdim=True)
    probs = torch.softmax(scores.masked_fill(empty, 0), -1).masked_fill(empty, 0)
    return probs @ v


def make_mask(mode, b, m, n, device, generator):
    if mode == "bool":
        mask = torch.rand(b, 1, m, n, device=device, generator=generator) > 0.3
        mask[:, :, 0, :] = False  # fully masked row -> zero output by definition
        return mask
    if mode == "additive":
        mask = torch.randn(m, n, device=device, generator=generator) * 0.2
        mask[0, :] = -float("inf")
        return mask
    return None


def parity_case(name, geom, dtype, device, output):
    _, b, h, hk, m, n, d, mode = geom
    generator = torch.Generator(device=device).manual_seed(fixtures.SEED)
    master = [
        torch.randn(b, heads, length, d, device=device, generator=generator) * 0.5
        for heads, length in ((h, m), (hk, n), (hk, n))
    ]
    mask = make_mask(mode, b, m, n, device, generator)
    causal, gqa = mode == "causal", h != hk
    scale = 1.0 / d**0.5
    opts = dict(attn_mask=mask, is_causal=causal, scale=scale, enable_gqa=gqa)
    dy_master = torch.randn_like(master[0], generator=generator)

    cpu_inputs = [x.detach().cpu().double().requires_grad_() for x in master]
    cpu_mask = None if mask is None else mask.cpu()
    if cpu_mask is not None and cpu_mask.dtype != torch.bool:
        cpu_mask = cpu_mask.double()
    ref_out = cpu_reference(*cpu_inputs, cpu_mask, causal, scale, gqa)
    dy_cpu = dy_master.cpu().double()
    ref_grads = torch.autograd.grad(ref_out, cpu_inputs, dy_cpu)

    inputs = [x.to(dtype).detach().requires_grad_() for x in master]
    dy = dy_master.to(dtype)
    out = TILED(*inputs, **opts)
    grads = torch.autograd.grad(out, inputs, dy)
    math_inputs = [x.detach().clone().requires_grad_() for x in inputs]
    with sdpa_kernel(SDPBackend.MATH):
        math_out = F.scaled_dot_product_attention(*math_inputs, **opts)
        math_grads = torch.autograd.grad(math_out, math_inputs, dy)

    bound = fixtures.REL_L2_BOUNDS[str(dtype).split(".")[-1]]
    rtol, atol = fixtures.CLOSE_BOUNDS[str(dtype).split(".")[-1]]
    actuals = [out, *grads]
    expected = [ref_out, *ref_grads]
    math_values = [math_out, *math_grads]
    errors, math_errors, close_ok, finite = [], [], True, True
    for actual, exp, math_val in zip(actuals, expected, math_values):
        finite &= bool(actual.isfinite().all())
        errors.append(rel_l2(actual.cpu(), exp.cpu()))
        math_errors.append(rel_l2(math_val.cpu(), exp.cpu()))
        try:
            torch.testing.assert_close(actual, math_val, rtol=rtol, atol=atol)
        except AssertionError:
            close_ok = False
    worst = max(errors)
    policy = fixtures.policy_for(str(dtype).split(".")[-1], d, True)
    check(
        f"parity-{name}-{str(dtype).split('.')[-1]}",
        finite and close_ok and worst <= bound,
        geometry=[b, h, hk, m, n, d],
        mode=mode,
        policy=policy,
        rel_l2=errors,
        rel_l2_bound=bound,
        math_rel_l2=math_errors,
        close_ok=close_ok,
        finite=finite,
    )
    del master, inputs, math_inputs, cpu_inputs, out, grads, math_out, math_grads
    del ref_out, ref_grads, expected, math_values, actuals
    gc.collect()
    torch.cuda.empty_cache()
    save(output)


def refusals(device, output):
    q = torch.randn(1, 2, 32, 64, device=device, requires_grad=True)
    k = torch.randn(1, 2, 32, 64, device=device, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    dy = torch.randn_like(q)

    exception_types = {
        "NotImplementedError": NotImplementedError,
        "ValueError": ValueError,
        "RuntimeError": RuntimeError,
        "Exception": Exception,
    }

    def attempt(name, expected, fn):
        try:
            fn()
        except Exception as error:
            check(
                f"refuse-{name}",
                isinstance(error, exception_types[expected]),
                expected=expected,
                actual=type(error).__name__,
            )
        else:
            check(f"refuse-{name}", False, expected=expected, actual="no-exception")
        save(output)

    attempt("dropout", "NotImplementedError", lambda: TILED(q, k, v, dropout_p=0.1))
    attempt(
        "mask-gradient",
        "NotImplementedError",
        lambda: TILED(
            q, k, v, attn_mask=torch.zeros(32, 32, device=device, requires_grad=True)
        ),
    )
    attempt(
        "causal-plus-mask",
        "ValueError",
        lambda: TILED(
            q,
            k,
            v,
            is_causal=True,
            attn_mask=torch.ones(32, 32, device=device, dtype=torch.bool),
        ),
    )
    attempt(
        "gqa-without-flag",
        "ValueError",
        lambda: TILED(q, k[:, :1], v[:, :1]),
    )
    attempt("nonfinite-scale", "ValueError", lambda: TILED(q, k, v, scale=float("nan")))
    attempt(
        "head-dim-257",
        "ValueError",
        lambda: TILED(
            torch.randn(1, 1, 8, 257, device=device),
            torch.randn(1, 1, 8, 257, device=device),
            torch.randn(1, 1, 8, 257, device=device),
        ),
    )
    attempt(
        "seq-4097",
        "ValueError",
        lambda: TILED(
            torch.randn(1, 1, 4097, 16, device=device),
            torch.randn(1, 1, 4097, 16, device=device),
            torch.randn(1, 1, 4097, 16, device=device),
        ),
    )
    attempt(
        "dtype-float64",
        "ValueError",
        lambda: TILED(q.double(), k.double(), v.double()),
    )
    attempt("dtype-mismatch", "ValueError", lambda: TILED(q.half(), k, v))
    attempt(
        "cpu-device",
        "ValueError",
        lambda: TILED(q.cpu(), k.cpu(), v.cpu()),
    )
    attempt(
        "wrong-rank",
        "ValueError",
        lambda: TILED(q[0], k[0], v[0]),
    )

    def higher_order():
        # once_differentiable returns detached grads on the create_graph call;
        # the refusal surfaces when differentiating those grads again.
        out = TILED(q, k, v)
        (dq,) = torch.autograd.grad(out, (q,), dy, create_graph=True)
        torch.autograd.grad(dq.sum(), q)

    attempt("higher-order-grad", "RuntimeError", higher_order)

    def compiled():
        fn = torch.compile(
            lambda a, b, c: TILED(a, b, c), fullgraph=True, dynamic=False
        )
        fn(q, k, v)

    attempt("compile-fullgraph", "Exception", compiled)


def checkpoint_parity(device, output):
    for dtype in (torch.float16, torch.bfloat16):
        inputs = [
            torch.randn(1, 4, 512, 64, device=device, dtype=dtype, requires_grad=True)
            for _ in range(3)
        ]

        def fn(*xs):
            return TILED(*xs, is_causal=True)

        out = checkpoint(fn, *inputs, use_reentrant=False)
        out.square().mean().backward()
        ck_grads = [x.grad.detach().clone() for x in inputs]
        for x in inputs:
            x.grad = None
        out2 = fn(*inputs)
        out2.square().mean().backward()
        bound = fixtures.REL_L2_BOUNDS[str(dtype).split(".")[-1]]
        errors = [rel_l2(a, x.grad) for a, x in zip(ck_grads, inputs)]
        check(
            f"checkpoint-{str(dtype).split('.')[-1]}",
            all(x.grad.isfinite().all() for x in inputs) and max(errors) <= bound,
            rel_l2=errors,
            rel_l2_bound=bound,
        )
        del inputs, out, out2, ck_grads
        gc.collect()
        torch.cuda.empty_cache()
        save(output)


def transformers_integration(device, output):
    from transformers import LlamaConfig, LlamaForCausalLM

    tiled_mod.register_transformers_attention()
    ids = torch.arange(256).remainder(127).reshape(2, 128)
    padding = torch.ones_like(ids)
    padding[0, :5] = 0
    for dtype, rtol, atol in (
        (torch.float32, 2e-3, 3e-5),
        (torch.bfloat16, 5e-2, 3e-2),
    ):
        cfg = LlamaConfig(
            vocab_size=127,
            hidden_size=256,
            intermediate_size=384,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            attention_dropout=0.0,
        )
        cfg._attn_implementation = "bc250_tiled"
        model = LlamaForCausalLM(cfg).to(device=device, dtype=dtype).train()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        cfg_cpu = LlamaConfig(**cfg.to_dict())
        cfg_cpu._attn_implementation = "sdpa"
        cpu_model = LlamaForCausalLM(cfg_cpu).double().train()
        cpu_model.load_state_dict(model.state_dict())
        result = model(
            input_ids=ids.to(device),
            attention_mask=padding.to(device),
            labels=ids.to(device),
            use_cache=False,
        )
        result.loss.backward()
        with sdpa_kernel(SDPBackend.MATH):
            cpu_result = cpu_model(
                input_ids=ids, attention_mask=padding, labels=ids, use_cache=False
            )
            cpu_result.loss.backward()
        finite = bool(result.logits.isfinite().all())
        ok = finite
        detail = {"loss": result.loss.item(), "reference_loss": cpu_result.loss.item()}
        try:
            torch.testing.assert_close(
                result.logits.cpu().double(), cpu_result.logits, rtol=rtol, atol=atol
            )
        except AssertionError as error:
            ok = False
            detail["logits_error"] = str(error)[:400]
        grad_errors = []
        for (name, p), (_, cp) in zip(
            model.named_parameters(), cpu_model.named_parameters()
        ):
            if p.grad is None:
                continue
            if cp.grad is None:
                ok = False
                detail.setdefault("missing_reference_grads", []).append(name)
                continue
            grad_errors.append(rel_l2(p.grad.cpu(), cp.grad))
            try:
                torch.testing.assert_close(
                    p.grad.cpu().double(), cp.grad, rtol=rtol, atol=atol, msg=name
                )
            except AssertionError:
                ok = False
                detail.setdefault("failed_grads", []).append(name)
        detail["worst_grad_rel_l2"] = max(grad_errors, default=0.0)
        check(f"transformers-{str(dtype).split('.')[-1]}", ok, **detail)
        del model, cpu_model, result, cpu_result
        gc.collect()
        torch.cuda.empty_cache()
        save(output)


def autocast_parity(device, output):
    for target in (torch.float16, torch.bfloat16):
        inputs = [
            torch.randn(1, 2, 256, 64, device=device, requires_grad=True)
            for _ in range(3)
        ]
        dy = torch.randn(1, 2, 256, 64, device=device)
        with torch.autocast("cuda", dtype=target):
            out = TILED(*inputs, is_causal=True)
            with sdpa_kernel(SDPBackend.MATH):
                math_out = F.scaled_dot_product_attention(
                    *[x.detach().clone().requires_grad_() for x in inputs],
                    is_causal=True,
                )
        grads = torch.autograd.grad(out, inputs, dy.to(target))
        rtol, atol = fixtures.CLOSE_BOUNDS[str(target).split(".")[-1]]
        ok = out.dtype == target and all(g.isfinite().all() for g in grads)
        try:
            torch.testing.assert_close(out, math_out, rtol=rtol, atol=atol)
        except AssertionError:
            ok = False
        check(
            f"autocast-{str(target).split('.')[-1]}",
            ok,
            output_dtype=str(out.dtype),
            grad_finite=all(g.isfinite().all() for g in grads),
        )
        del inputs, out, math_out, grads
        gc.collect()
        torch.cuda.empty_cache()
        save(output)


def empty_dimensions(device, output):
    for m, n in ((3, 0), (0, 3)):
        xs = [
            torch.randn(1, 1, length, 16, device=device, requires_grad=True)
            for length in (m, n, n)
        ]
        y = TILED(*xs)
        y.sum().backward()
        check(
            f"empty-m{m}-n{n}",
            torch.count_nonzero(y).item() == 0
            and all(torch.count_nonzero(x.grad).item() == 0 for x in xs),
        )
        del xs, y
        save(output)


def graph_probe(device, output):
    """Informational only: record whether CUDA-graph capture works; not gated."""
    xs = [torch.randn(1, 2, 128, 64, device=device) for _ in range(3)]
    eager = TILED(*xs, is_causal=True)
    detail = {}
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                TILED(*xs, is_causal=True)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = TILED(*xs, is_causal=True)
        graph.replay()
        torch.cuda.synchronize()
        detail = {
            "captured": True,
            "replay_rel_l2": rel_l2(captured, eager),
        }
        del graph
    except Exception as error:
        detail = {"captured": False, "error": repr(error)[:400]}
    check("graph-capture-probe-informational", True, **detail)
    del xs, eager
    gc.collect()
    torch.cuda.empty_cache()
    save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--part", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    REPORT["part"] = args.part
    REPORT["status"] = "running"
    torch.set_num_threads(4)
    torch.manual_seed(fixtures.SEED)
    device = torch.device("cuda")
    env = REPORT["environment"]
    env["torch"] = torch.__version__
    env["hip"] = torch.version.hip
    env["gcnArchName"] = torch.cuda.get_device_properties(device).gcnArchName
    env["tiled_module"] = str(Path(tiled_mod.__file__).resolve())
    env["tiled_module_sha256"] = hashlib.sha256(
        Path(tiled_mod.__file__).read_bytes()
    ).hexdigest()
    check(
        "gfx1013-native",
        env["gcnArchName"].split(":")[0] == "gfx1013",
        arch=env["gcnArchName"],
    )
    save(args.output)
    for geom in fixtures.parity_geometries(args.part):
        for dtype_name in fixtures.DTYPES:
            parity_case(geom[0], geom, getattr(torch, dtype_name), device, args.output)
    if args.part == 1:
        refusals(device, args.output)
        checkpoint_parity(device, args.output)
        transformers_integration(device, args.output)
        autocast_parity(device, args.output)
        empty_dimensions(device, args.output)
        graph_probe(device, args.output)
    REPORT["status"] = "passed" if REPORT["summary"]["all_passed"] else "failed"
    save(args.output)
    print(
        f"ATTN_QUALIFY part={args.part} "
        f"{REPORT['summary']['passed']}/{REPORT['summary']['total']} "
        f"status={REPORT['status']}",
        flush=True,
    )
    return 0 if REPORT["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
