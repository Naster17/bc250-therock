#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PLAN2-05 Part B2: fresh-process graph recapture/resume qualification.

Phases run as separate processes inside the one guardian child:
  ref    — eager 16 updates, save trajectory + final trainable/optimizer state.
  graph  — graphed 8 updates, save the 8-update checkpoint state + trajectory.
  resume — FRESH process loads the 8-update state, RECAPTURES, runs 8 more.
Parent compares the combined graph+resume 16-update trajectory and final state
against the single-process eager reference (rel-L2 <= 1e-5 FP32). This proves
the resume boundary: the graph pool is not checkpointable, so resume requires a
fresh recapture from loaded state, which reproduces the eager trajectory.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from compile_graph_plan2_05 import fixtures, models  # noqa: E402
from compile_graph_plan2_05.graph_step import GraphedStep  # noqa: E402

UPDATES = fixtures.RESUME_UPDATES
HALF = UPDATES // 2
BOUND = fixtures.GRAPH_FP32_BOUND


def rel_l2(a, b):
    x, y = a.detach().double(), b.detach().double()
    return float((x - y).norm() / y.norm().clamp_min(1e-30))


def build(device):
    torch.manual_seed(fixtures.SEED)
    model = (
        models.LoRAStack(
            fixtures.STACK_LAYERS, fixtures.STACK_WIDTH, fixtures.STACK_RANK
        )
        .to(device=device, dtype=torch.float32)
        .train()
    )
    generator = torch.Generator(device=device).manual_seed(fixtures.SEED + 1)
    x = torch.randn(
        fixtures.STACK_BATCH,
        fixtures.STACK_SEQ,
        fixtures.STACK_WIDTH,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    target = torch.randn_like(x, generator=generator)
    opt = torch.optim.AdamW(models.trainable_parameters(model), lr=1e-3)
    return model, x, target, opt


def trainable_state(model):
    return {
        n: p.detach().cpu().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }


def load_trainable(model, state):
    own = {n: p for n, p in model.named_parameters() if p.requires_grad}
    for n, t in state.items():
        own[n].data.copy_(t.to(own[n].device))


def worst_rel(actual, reference):
    keys = set(actual) & set(reference)
    return max((rel_l2(actual[k], reference[k]) for k in keys), default=float("inf"))


def traj_rel(actual, reference):
    num = sum((a - b) ** 2 for a, b in zip(actual, reference)) ** 0.5
    den = sum(b * b for b in reference) ** 0.5
    return num / max(den, 1e-30)


def phase_ref(device, state_dir):
    model, x, target, opt = build(device)
    traj = []
    for _ in range(UPDATES):
        model.zero_grad(set_to_none=False)
        loss = model.loss(x, target)
        loss.backward()
        opt.step()
        traj.append(float(loss.item()))
    torch.save(
        {"traj": traj, "weights": trainable_state(model), "opt": opt.state_dict()},
        state_dir / "ref.pt",
    )


def phase_graph(device, state_dir):
    model, x, target, opt = build(device)
    loss_buf = torch.zeros((), device=device)

    def step():
        model.zero_grad(set_to_none=False)
        loss = model.loss(x, target)
        loss_buf.copy_(loss.detach())
        loss.backward()

    graphed = GraphedStep(step, warmup=3).capture()
    traj = []
    for _ in range(HALF):
        graphed.replay()
        opt.step()
        torch.cuda.synchronize()
        traj.append(float(loss_buf.item()))
    torch.save(
        {"traj": traj, "weights": trainable_state(model), "opt": opt.state_dict()},
        state_dir / "graph8.pt",
    )


def phase_resume(device, state_dir):
    ckpt = torch.load(state_dir / "graph8.pt", weights_only=False)
    model, x, target, opt = build(device)
    load_trainable(model, ckpt["weights"])
    opt.load_state_dict(ckpt["opt"])
    loss_buf = torch.zeros((), device=device)

    def step():
        model.zero_grad(set_to_none=False)
        loss = model.loss(x, target)
        loss_buf.copy_(loss.detach())
        loss.backward()

    graphed = GraphedStep(step, warmup=3).capture()  # fresh recapture
    traj = []
    for _ in range(HALF):
        graphed.replay()
        opt.step()
        torch.cuda.synchronize()
        traj.append(float(loss_buf.item()))
    torch.save(
        {"traj": traj, "weights": trainable_state(model), "opt": opt.state_dict()},
        state_dir / "resume8.pt",
    )


def opt_named(model, opt):
    pid_to_name = {id(p): n for n, p in model.named_parameters()}
    out = {}
    for param, state in opt.state.items():
        name = pid_to_name.get(id(param))
        if name is None:
            continue
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state:
                out[f"{name}.{key}"] = state[key].detach().cpu().clone()
    return out


def parent(device, state_dir, output):
    checks = []
    for phase in ("ref", "graph", "resume"):
        proc = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--phase",
                phase,
                "--state-dir",
                str(state_dir),
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            checks.append(
                {"name": f"phase-{phase}", "passed": False, "error": proc.stderr[-800:]}
            )
            _write(output, checks)
            return 1
        checks.append({"name": f"phase-{phase}", "passed": True})
    ref = torch.load(state_dir / "ref.pt", weights_only=False)
    g8 = torch.load(state_dir / "graph8.pt", weights_only=False)
    r8 = torch.load(state_dir / "resume8.pt", weights_only=False)
    combined_traj = g8["traj"] + r8["traj"]
    t_rel = traj_rel(combined_traj, ref["traj"])
    w_rel = worst_rel(r8["weights"], ref["weights"])
    # Optimizer state comparison via a fresh model to map names.
    model, _, _, opt = build(device)
    load_trainable(model, r8["weights"])
    opt.load_state_dict(r8["opt"])
    got_opt = opt_named(model, opt)
    ref_model, _, _, ref_opt = build(device)
    load_trainable(ref_model, ref["weights"])
    ref_opt.load_state_dict(ref["opt"])
    ref_opt_named = opt_named(ref_model, ref_opt)
    o_rel = worst_rel(got_opt, ref_opt_named)
    checks.append(
        {
            "name": "fresh-recapture-resume-parity",
            "passed": t_rel <= BOUND and w_rel <= BOUND and o_rel <= BOUND,
            "trajectory_rel_l2": t_rel,
            "adapter_rel_l2": w_rel,
            "optimizer_state_rel_l2": o_rel,
            "bound": BOUND,
            "updates": UPDATES,
        }
    )
    return _write(output, checks)


def _write(output, checks):
    summary = {"total": len(checks), "passed": sum(c["passed"] for c in checks)}
    summary["all_passed"] = summary["passed"] == len(checks)
    report = {
        "status": "passed" if summary["all_passed"] else "failed",
        "checks": checks,
        "summary": summary,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"CG_RESUME {summary['passed']}/{summary['total']} "
        f"status={report['status']}",
        flush=True,
    )
    return 0 if summary["all_passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument(
        "--phase", choices=("parent", "ref", "graph", "resume"), default="parent"
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda")
    if args.phase == "parent":
        args.state_dir.mkdir(parents=True, exist_ok=True)
        return parent(device, args.state_dir, args.output)
    {"ref": phase_ref, "graph": phase_graph, "resume": phase_resume}[args.phase](
        device, args.state_dir
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
