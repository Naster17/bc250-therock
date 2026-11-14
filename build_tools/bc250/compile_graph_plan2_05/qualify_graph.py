#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PLAN2-05 Part B: graph-capture integration GPU qualification (in guardian).

Owned LoRA-pattern stack: multi-update parity of graphed fwd+bwd + eager AdamW,
whole-step capturable AdamW, and checkpointing + graphed fwd+bwd against an
eager reference from identical init; whole-step timing; graph-safety refusals;
make_graphed_callables forward parity. FP32 tight (1e-5), FP16/BF16 looser.
No source/default change.
"""

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from compile_graph_plan2_05 import fixtures, models  # noqa: E402
from compile_graph_plan2_05.graph_step import (  # noqa: E402
    GraphCaptureError,
    GraphedStep,
    assert_graph_safe,
    capture_graphed_callable,
)

REPORT = {"checks": [], "measurements": {}, "environment": {}, "part": None}
STARTED = time.monotonic()


def check(name, passed, **detail):
    REPORT["checks"].append({"name": name, "passed": bool(passed), **detail})
    if not passed:
        print(f"CG_FAIL {name} {detail}", flush=True)
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


def rel_l2(a, b):
    x, y = a.detach().double(), b.detach().double()
    return float((x - y).norm() / y.norm().clamp_min(1e-30))


def build_stack(device, dtype, checkpointing=False, ac=None):
    # With autocast (ac), parameters/inputs stay FP32 masters and the forward
    # runs in ac — the qualified D26-D31 low-precision training policy.
    master = torch.float32 if ac is not None else dtype
    torch.manual_seed(fixtures.SEED)
    model = (
        models.LoRAStack(
            fixtures.STACK_LAYERS,
            fixtures.STACK_WIDTH,
            fixtures.STACK_RANK,
            checkpointing=checkpointing,
        )
        .to(device=device, dtype=master)
        .train()
    )
    generator = torch.Generator(device=device).manual_seed(fixtures.SEED + 1)
    x = torch.randn(
        fixtures.STACK_BATCH,
        fixtures.STACK_SEQ,
        fixtures.STACK_WIDTH,
        device=device,
        dtype=master,
        generator=generator,
    )
    target = torch.randn_like(x, generator=generator)
    return model, x, target


def compute_loss(model, x, target, ac):
    if ac is not None:
        with torch.autocast("cuda", dtype=ac):
            return model.loss(x, target)
    return model.loss(x, target)


def make_step(model, opt, x, target, include_opt, loss_buf, ac=None):
    def step():
        model.zero_grad(set_to_none=False)
        loss = compute_loss(model, x, target, ac)
        loss_buf.copy_(loss.detach())
        loss.backward()
        if include_opt:
            opt.step()

    return step


def trainable_state(model):
    return {
        n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad
    }


def opt_state_by_name(model, opt):
    pid_to_name = {id(p): n for n, p in model.named_parameters()}
    out = {}
    for param, state in opt.state.items():
        name = pid_to_name.get(id(param))
        if name is None:
            continue
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state:
                out[f"{name}.{key}"] = state[key].detach().clone()
    return out


def worst_rel(actual, reference):
    keys = set(actual) & set(reference)
    if not keys:
        return float("inf")
    return max(rel_l2(actual[k], reference[k]) for k in keys)


def traj_rel(actual, reference):
    if len(actual) != len(reference):
        return float("inf")
    num = sum((a - b) ** 2 for a, b in zip(actual, reference)) ** 0.5
    den = sum(b * b for b in reference) ** 0.5
    return num / max(den, 1e-30)


def run_eager(
    device, dtype, checkpointing, updates, ac=None, foreach=None, capturable=False
):
    model, x, target = build_stack(device, dtype, checkpointing, ac)
    opt = torch.optim.AdamW(
        models.trainable_parameters(model),
        lr=1e-3,
        foreach=foreach,
        capturable=capturable,
    )
    traj = []
    for _ in range(updates):
        model.zero_grad(set_to_none=False)
        loss = compute_loss(model, x, target, ac)
        loss.backward()
        opt.step()
        traj.append(float(loss.item()))
    return traj, trainable_state(model), opt_state_by_name(model, opt)


def run_graphed(device, dtype, checkpointing, updates, whole_step, ac=None):
    model, x, target = build_stack(device, dtype, checkpointing, ac)
    init_weights = trainable_state(model)
    opt = torch.optim.AdamW(
        models.trainable_parameters(model),
        lr=1e-3,
        capturable=whole_step,
        foreach=False if whole_step else None,
    )
    loss_buf = torch.zeros((), device=device, dtype=torch.float32)
    step = make_step(model, opt, x, target, whole_step, loss_buf, ac)
    graphed = GraphedStep(step, warmup=3).capture()
    if whole_step:
        # Warmup ran opt.step() inside the closure, advancing weights/state by 3
        # steps. Reset the static weight + optimizer-state tensors in place to the
        # initial state so the replayed trajectory starts where eager does.
        own = {n: p for n, p in model.named_parameters() if p.requires_grad}
        for n, t in init_weights.items():
            own[n].data.copy_(t)
        for state in opt.state.values():
            for key in ("exp_avg", "exp_avg_sq", "step"):
                if key in state:
                    state[key].zero_()
    traj = []
    for _ in range(updates):
        graphed.replay()
        if not whole_step:
            opt.step()
        torch.cuda.synchronize()
        traj.append(float(loss_buf.item()))
    return traj, trainable_state(model), opt_state_by_name(model, opt)


def parity_case(label, device, dtype_name, checkpointing, whole_step, output, ac=None):
    dtype = getattr(torch, dtype_name)
    bound = fixtures.graph_bound(dtype_name)
    updates = fixtures.GRAPH_UPDATES
    # Match the optimizer variant exactly: whole-step capture uses capturable
    # foreach=False AdamW, so the eager reference uses the same (device-side step
    # arithmetic); otherwise capturable-vs-not differs at ~1e-5 in optimizer state.
    ref_traj, ref_w, ref_opt = run_eager(
        device,
        dtype,
        checkpointing,
        updates,
        ac,
        foreach=False if whole_step else None,
        capturable=whole_step,
    )
    got_traj, got_w, got_opt = run_graphed(
        device, dtype, checkpointing, updates, whole_step, ac
    )
    t_rel = traj_rel(got_traj, ref_traj)
    w_rel = worst_rel(got_w, ref_w)
    o_rel = worst_rel(got_opt, ref_opt)
    ok = t_rel <= bound and w_rel <= bound and o_rel <= bound
    check(
        f"graph-{label}-{dtype_name}",
        ok,
        trajectory_rel_l2=t_rel,
        adapter_rel_l2=w_rel,
        optimizer_state_rel_l2=o_rel,
        bound=bound,
        ref_loss_first_last=[ref_traj[0], ref_traj[-1]],
        got_loss_first_last=[got_traj[0], got_traj[-1]],
    )
    gc.collect()
    torch.cuda.empty_cache()
    save(output)


def run_parity(device, output):
    parity_case("fwd_bwd-eager_opt", device, "float32", False, False, output)
    parity_case("whole_step", device, "float32", False, True, output)
    parity_case("checkpoint-fwd_bwd", device, "float32", True, False, output)
    parity_case(
        "fwd_bwd-eager_opt", device, "float16", False, False, output, ac=torch.float16
    )
    parity_case(
        "fwd_bwd-eager_opt", device, "bfloat16", False, False, output, ac=torch.bfloat16
    )
    timing(device, output)
    graphed_callables(device, output)


def time_fn(fn, warmup=fixtures.WARMUPS, samples=fixtures.SAMPLES):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    vals = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        vals.append(start.elapsed_time(end))
    return statistics.median(vals)


def timing(device, output):
    dtype = torch.float32
    model, x, target = build_stack(device, dtype)
    opt = torch.optim.AdamW(models.trainable_parameters(model), lr=1e-3)
    loss_buf = torch.zeros((), device=device, dtype=dtype)

    def eager_step():
        model.zero_grad(set_to_none=False)
        loss = model.loss(x, target)
        loss.backward()
        opt.step()

    eager_ms = time_fn(eager_step)
    gmodel, gx, gtarget = build_stack(device, dtype)
    gopt = torch.optim.AdamW(models.trainable_parameters(gmodel), lr=1e-3)
    gbuf = torch.zeros((), device=device, dtype=dtype)
    gstep = make_step(gmodel, gopt, gx, gtarget, False, gbuf)
    graphed = GraphedStep(gstep, warmup=3).capture()

    def graph_step():
        graphed.replay()
        gopt.step()

    graph_ms = time_fn(graph_step)
    REPORT["measurements"]["step_timing"] = {
        "eager_ms": eager_ms,
        "graphed_ms": graph_ms,
        "speedup_eager_over_graphed": eager_ms / graph_ms,
    }
    check(
        "graph-step-timing-recorded",
        graph_ms > 0,
        eager_ms=eager_ms,
        graphed_ms=graph_ms,
        speedup=eager_ms / graph_ms,
    )
    del model, gmodel, graphed
    gc.collect()
    torch.cuda.empty_cache()
    save(output)


def graphed_callables(device, output):
    dtype = torch.float32
    model, x, _ = build_stack(device, dtype)
    eager_out = model(x)
    ref = eager_out.detach().clone()
    eager_loss = eager_out.square().mean()
    eager_loss.backward()
    ref_grads = models.grad_snapshot(model)
    del eager_out, eager_loss
    model.zero_grad(set_to_none=True)
    graphed = capture_graphed_callable(model, (x,), allow_unused_input=True)
    out = graphed(x)
    out.square().mean().backward()
    torch.cuda.synchronize()
    o_rel = rel_l2(out, ref)
    g_rel = worst_rel(models.grad_snapshot(model), ref_grads)
    check(
        "graphed_callables-forward-backward",
        o_rel <= 1e-5 and g_rel <= 1e-5,
        output_rel_l2=o_rel,
        grad_rel_l2=g_rel,
    )
    del model, graphed
    gc.collect()
    torch.cuda.empty_cache()
    save(output)


def run_refusals(device, output):
    model = models.LoRAStack(4, 32, 4).to(device).train()
    model.blocks[0].dropout = torch.nn.Dropout(p=0.1)
    x = torch.randn(2, 8, 32, device=device)
    try:
        assert_graph_safe(model, (x,))
        check("refuse-dropout", False, actual="no-exception")
    except GraphCaptureError:
        check("refuse-dropout", True)
    stale = torch.randn(2, 8, 32, device=device, requires_grad=True)
    stale.square().sum().backward()
    clean = models.LoRAStack(4, 32, 4).to(device).train()
    try:
        assert_graph_safe(clean, (stale,))
        check("refuse-stale-eager-graph", False, actual="no-exception")
    except GraphCaptureError:
        check("refuse-stale-eager-graph", True)
    save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--part", type=int, choices=(1,), required=True)
    args = parser.parse_args()
    REPORT["part"] = args.part
    REPORT["status"] = "running"
    torch.set_num_threads(4)
    device = torch.device("cuda")
    env = REPORT["environment"]
    env["torch"] = torch.__version__
    env["hip"] = torch.version.hip
    env["gcnArchName"] = torch.cuda.get_device_properties(device).gcnArchName
    check(
        "gfx1013-native",
        env["gcnArchName"].split(":")[0] == "gfx1013",
        arch=env["gcnArchName"],
    )
    save(args.output)
    run_parity(device, args.output)
    run_refusals(device, args.output)
    REPORT["status"] = "passed" if REPORT["summary"]["all_passed"] else "failed"
    save(args.output)
    print(
        f"CG_GRAPH part={args.part} "
        f"{REPORT['summary']['passed']}/{REPORT['summary']['total']} "
        f"status={REPORT['status']}",
        flush=True,
    )
    return 0 if REPORT["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
