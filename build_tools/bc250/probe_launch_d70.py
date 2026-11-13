#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""D70 launch-reduction feasibility probe (owned tiny fixtures, no app).

Sections (each produces a verdict; "blocked" is a finding, not a failure):
    profiler                 torch.profiler CUDA activity on this port
    graph_mlp                HIP graph capture of a launch-bound MLP fwd+bwd
    graph_qwen35_aten        tiny Qwen3.5 fwd+bwd capture (installed ATen path)
    graph_qwen35_fused       same with the D68 fused route (Triton in capture)
    graph_checkpoint         capture with gradient checkpointing enabled
    graphed_callables        torch.cuda.make_graphed_callables integration path
    instrumentation_selftest D70 span instrumentation on the tiny model with
                             checkpointing (hook pairing + budget closure)

Pre-declared bounds: graph replay parity rel-L2 <= 1e-5 (FP32); selftest GPU
timeline coverage within 1 ± 10% and CPU wall closure residual <= 1e-6 s
(event overhead is relatively large at this size). The fixture exits 0 when
every section produced a verdict and none FAILED (parity/fixture errors);
blocked captures exit 0 with the blocker recorded. Run under the D67-pattern
guardian in the app environment.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time

FIXTURE = "probe_launch_d70"
GRAPH_PARITY_REL_L2 = 1e-5
SELFTEST_RESIDUAL_FRACTION = 0.10
TIMING_ROUNDS = 20
MLP_LAYERS = 64
MLP_WIDTH = 64
MLP_BATCH = 8
MODEL_SEQ = 64
MODEL_BATCH = 2


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


class ParityError(ValueError):
    """A successful capture replayed wrong numbers — fixture failure, not a
    feasibility blocker."""


def rel_l2(actual, reference) -> float:
    a = actual.detach().double()
    b = reference.detach().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def verdict(section, status, **fields):
    section.update(status=status, **fields)
    print(
        f"D70_PROBE {section['name']} {status} "
        f"{json.dumps({k: v for k, v in fields.items() if not isinstance(v, (list, dict))})}",
        flush=True,
    )


def build_mlp(torch, device):
    layers = []
    for _ in range(MLP_LAYERS):
        layers.append(torch.nn.Linear(MLP_WIDTH, MLP_WIDTH))
        layers.append(torch.nn.GELU())
    return torch.nn.Sequential(*layers).to(device).train()


def mlp_step(torch, model, static_in):
    model.zero_grad(set_to_none=False)
    out = model(static_in)
    loss = out.square().mean()
    loss.backward()
    return loss


def capture(torch, fn, warmup=3):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def time_fn(torch, fn, rounds=TIMING_ROUNDS):
    samples = []
    for _ in range(rounds):
        torch.cuda.synchronize()
        t0 = time.monotonic()
        fn()
        torch.cuda.synchronize()
        samples.append((time.monotonic() - t0) * 1000.0)
    return statistics.median(samples)


def grad_snapshot(model):
    return {
        n: p.grad.detach().clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }


def worst_grad_rel(actual, reference):
    names = set(reference) & set(actual)
    if not names:
        raise ValueError("no overlapping gradients")
    return max(rel_l2(actual[n], reference[n]) for n in names)


def build_tiny(torch, device):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    config = Qwen3_5TextConfig(**tiny_config_overrides())
    # The app loads with attn_implementation="eager"; match it exactly.
    config._attn_implementation = "eager"
    model = Qwen3_5ForCausalLM(config)
    return model.to(device).train()


def tiny_step(torch, model, ids):
    model.zero_grad(set_to_none=False)
    out = model(input_ids=ids, labels=ids, use_cache=False)
    out.loss.backward()
    return out.loss


def graph_parity_and_timing(torch, section, model, step_fn, eager_loss, eager_grads):
    """Capture step_fn, check replay parity against the eager snapshot, time
    eager vs replay. Raises on parity violation (fixture failure)."""
    graph = capture(torch, step_fn)
    graph.replay()
    torch.cuda.synchronize()
    loss_rel = rel_l2(step_fn.last_loss, eager_loss)
    grad_rel = worst_grad_rel(grad_snapshot(model), eager_grads)
    if loss_rel > GRAPH_PARITY_REL_L2 or grad_rel > GRAPH_PARITY_REL_L2:
        raise ParityError(
            f"replay parity loss {loss_rel:.3e} grad {grad_rel:.3e} "
            f"> {GRAPH_PARITY_REL_L2:.3e}"
        )
    eager_ms = time_fn(torch, step_fn)
    replay_ms = time_fn(torch, graph.replay)
    verdict(
        section,
        "works",
        loss_rel_l2=loss_rel,
        max_grad_rel_l2=grad_rel,
        eager_ms=eager_ms,
        replay_ms=replay_ms,
        speedup=eager_ms / replay_ms,
    )


class LossRecorder:
    """step_fn wrapper keeping the last loss for parity comparisons."""

    def __init__(self, fn):
        self.fn = fn
        self.last_loss = None

    def __call__(self):
        self.last_loss = self.fn().detach().clone()


def section_profiler(torch, report, device):
    section = {"name": "profiler"}
    report["sections"].append(section)
    try:
        from torch.profiler import ProfilerActivity, profile

        torch.manual_seed(7001)
        model = build_mlp(torch, device)
        static_in = torch.randn(MLP_BATCH, MLP_WIDTH, device=device)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                mlp_step(torch, model, static_in)
        torch.cuda.synchronize()
        from torch.autograd import DeviceType

        # Kernel events only: parent aten:: ops carry the same device time
        # and would double-count it.
        kernels = [
            {
                "name": e.key,
                "count": e.count,
                "self_device_ms": e.self_device_time_total / 1000.0,
            }
            for e in prof.key_averages()
            if e.device_type == DeviceType.CUDA
        ]
        kernels.sort(key=lambda k: k["self_device_ms"], reverse=True)
        if not kernels:
            verdict(section, "blocked", error="no CUDA kernel events recorded")
            return
        verdict(
            section,
            "works",
            kernel_launches=sum(k["count"] for k in kernels),
            distinct_kernels=len(kernels),
            top_kernels=kernels[:10],
        )
    except Exception as exc:
        verdict(section, "blocked", error=f"{type(exc).__name__}: {exc}")


def section_graph_mlp(torch, report, device):
    section = {"name": "graph_mlp"}
    report["sections"].append(section)
    try:
        torch.manual_seed(7002)
        model = build_mlp(torch, device)
        static_in = torch.randn(MLP_BATCH, MLP_WIDTH, device=device)
        step = LossRecorder(lambda: mlp_step(torch, model, static_in))
        step()
        torch.cuda.synchronize()
        eager_loss = step.last_loss.clone()
        eager_grads = grad_snapshot(model)
        graph_parity_and_timing(torch, section, model, step, eager_loss, eager_grads)
    except Exception as exc:
        if section.get("status") != "works":
            status = "failed" if isinstance(exc, ParityError) else "blocked"
            verdict(section, status, error=f"{type(exc).__name__}: {exc}")


def _qwen35_graph_section(torch, report, device, name, candidate_path, checkpointing):
    section = {"name": name}
    report["sections"].append(section)
    record = None
    try:
        import importlib
        import transformers

        modeling = importlib.import_module(
            "transformers.models.qwen3_5.modeling_qwen3_5"
        )
        if candidate_path is not None:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import route_qwen35_gated_delta_d69 as routing

            record = routing.install(target="fused", candidate_path=candidate_path)
        torch.manual_seed(7003)
        model = build_tiny(torch, device)
        if checkpointing:
            model.gradient_checkpointing_enable()
        ids = torch.randint(0, 256, (MODEL_BATCH, MODEL_SEQ), device=device)
        step = LossRecorder(lambda: tiny_step(torch, model, ids))
        step()
        torch.cuda.synchronize()
        eager_loss = step.last_loss.clone()
        eager_grads = grad_snapshot(model)
        graph_parity_and_timing(torch, section, model, step, eager_loss, eager_grads)
        if record is not None:
            section["native_calls"] = record["native_calls"]
            section["fallbacks"] = record["fallbacks"]
    except Exception as exc:
        if section.get("status") != "works":
            status = "failed" if isinstance(exc, ParityError) else "blocked"
            verdict(section, status, error=f"{type(exc).__name__}: {exc}")
    finally:
        if record is not None:
            import route_qwen35_gated_delta_d69 as routing

            routing.uninstall(record)
        gc.collect()
        torch.cuda.empty_cache()


def section_graphed_callables(torch, report, device):
    section = {"name": "graphed_callables"}
    report["sections"].append(section)
    try:
        torch.manual_seed(7004)
        model = build_mlp(torch, device)
        static_in = torch.randn(MLP_BATCH, MLP_WIDTH, device=device)
        eager_out = model(static_in)
        eager_loss = eager_out.square().mean()
        eager_loss.backward()
        eager_grads = grad_snapshot(model)
        eager_out_ref = eager_out.detach().clone()
        # Keeping the eager autograd graph alive leaves stale AccumulateGrad
        # stream references that invalidate the later capture.
        del eager_out, eager_loss
        model.zero_grad(set_to_none=True)
        graphed = torch.cuda.make_graphed_callables(
            model, (static_in,), allow_unused_input=True
        )
        out = graphed(static_in)
        loss = out.square().mean()
        loss.backward()
        torch.cuda.synchronize()
        out_rel = rel_l2(out, eager_out_ref)
        grad_rel = worst_grad_rel(grad_snapshot(model), eager_grads)
        if out_rel > GRAPH_PARITY_REL_L2 or grad_rel > GRAPH_PARITY_REL_L2:
            raise ParityError(f"graphed parity out {out_rel:.3e} grad {grad_rel:.3e}")
        verdict(section, "works", out_rel_l2=out_rel, max_grad_rel_l2=grad_rel)
    except Exception as exc:
        if section.get("status") != "works":
            status = "failed" if isinstance(exc, ParityError) else "blocked"
            verdict(section, status, error=f"{type(exc).__name__}: {exc}")


def section_instrumentation_selftest(torch, report, device):
    section = {"name": "instrumentation_selftest"}
    report["sections"].append(section)
    handles, restores = [], []
    try:
        import importlib
        import decompose_lora_d70 as dec

        modeling = importlib.import_module(
            "transformers.models.qwen3_5.modeling_qwen3_5"
        )
        torch.manual_seed(7005)
        model = build_tiny(torch, device)
        model.gradient_checkpointing_enable()
        ids = torch.randint(0, 256, (MODEL_BATCH, MODEL_SEQ), device=device)
        recorder = dec.Recorder(torch, grad_accum=1)
        dec.wrap_functions(modeling, recorder, restores)
        layer_types = dec.install_instrumentation(model, recorder, handles)
        recorder.micro = 0
        recorder.phase = "forward"
        with recorder.span("micro_total"):
            out = model(input_ids=ids, labels=ids, use_cache=False)
            recorder.phase = "backward"
            with recorder.span("backward_total"):
                out.loss.backward()
            recorder.phase = "idle"
        with recorder.span("optimizer", update=0):
            pass
        recorder.update_completed = 1
        resolved = recorder.finalize()
        agg = dec.aggregate_update(resolved, 0, 1, layer_types)
        coverage = {
            cat: sum(1 for s in resolved if s["cat"] == cat)
            for cat in sorted({s["cat"] for s in resolved})
        }

        def phase_count(cat, phase):
            return sum(1 for s in resolved if s["cat"] == cat and s["phase"] == phase)

        # diag1 semantics: forward spans cover the original forward only
        # (recompute post hooks are interrupted); backward spans fire once.
        expected = {
            "layer/linear_attention": 6,  # 3 layers x (1 fwd + 1 bwd)
            "layer/full_attention": 2,  # 1 layer x (1 fwd + 1 bwd)
            "mod/mlp": 8,  # 4 layers x (1 fwd + 1 bwd)
        }
        mismatches = {
            cat: {"expected": n, "actual": coverage.get(cat, 0)}
            for cat, n in expected.items()
            if coverage.get(cat, 0) != n
        }
        for cat, n in (("fn/conv", 3), ("fn/chunk", 3), ("fn/eager", 1)):
            if phase_count(cat, "forward") != n:
                mismatches[cat + ":forward"] = {
                    "expected": n,
                    "actual": phase_count(cat, "forward"),
                }
        timeline_coverage = agg["gpu_timeline_coverage"]
        closure = abs(agg["cpu"]["closure_residual"])
        if mismatches or recorder.unpaired:
            raise ValueError(
                f"hook coverage mismatch {mismatches} unpaired={recorder.unpaired}"
            )
        if abs(timeline_coverage - 1.0) > SELFTEST_RESIDUAL_FRACTION:
            raise ValueError(
                f"gpu timeline coverage {timeline_coverage:.3f} outside "
                f"1 ± {SELFTEST_RESIDUAL_FRACTION}"
            )
        if closure > 1e-6:
            raise ValueError(f"cpu wall closure residual {closure:.3e}")
        dec.uninstall(restores, handles)
        handles, restores = [], []
        bwd_total = sum(s["seconds"] for s in resolved if s["cat"] == "backward_total")
        layer_bwd = sum(
            s["seconds"]
            for s in resolved
            if s["cat"].startswith("layer/") and s["phase"] == "backward"
        )
        verdict(
            section,
            "works",
            coverage=coverage,
            hook_fires=recorder.fires,
            pending_leftover=recorder.pending_leftover,
            gpu_timeline_coverage=timeline_coverage,
            cpu_closure_residual=closure,
            wall_seconds=agg["wall_seconds"],
            backward_total_seconds=bwd_total,
            layer_backward_seconds=layer_bwd,
            recompute_outside_layer_bwd_seconds=bwd_total - layer_bwd,
            fn_recompute_seconds={
                cat: phase_count(cat, "backward")
                for cat in ("fn/conv", "fn/chunk", "fn/eager")
            },
            shares=agg["shares"],
        )
    except Exception as exc:
        if section.get("status") != "works":
            verdict(section, "failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if handles or restores:
            import decompose_lora_d70 as dec

            try:
                dec.uninstall(restores, handles)
            except Exception:
                pass
        gc.collect()
        torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, default=None)
    args = parser.parse_args()

    import torch
    import transformers

    if not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm device required")
    device = torch.device("cuda")
    report = {
        "fixture": FIXTURE,
        "status": "running",
        "parity_bounds": {
            "graph_replay_rel_l2": GRAPH_PARITY_REL_L2,
            "selftest_residual_fraction": SELFTEST_RESIDUAL_FRACTION,
        },
        "environment": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "transformers": transformers.__version__,
            "device": torch.cuda.get_device_properties(0).gcnArchName,
            "device_name": torch.cuda.get_device_properties(0).name,
        },
        "sections": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        section_profiler(torch, report, device)
        section_graph_mlp(torch, report, device)
        _qwen35_graph_section(torch, report, device, "graph_qwen35_aten", None, False)
        if args.candidate is not None:
            _qwen35_graph_section(
                torch, report, device, "graph_qwen35_fused", str(args.candidate), False
            )
        else:
            report["sections"].append(
                {
                    "name": "graph_qwen35_fused",
                    "status": "skipped",
                    "error": "no candidate staged",
                }
            )
        _qwen35_graph_section(torch, report, device, "graph_checkpoint", None, True)
        section_graphed_callables(torch, report, device)
        section_instrumentation_selftest(torch, report, device)
        failed = [s["name"] for s in report["sections"] if s.get("status") == "failed"]
        report["status"] = "completed" if not failed else "failed"
        report["failed_sections"] = failed
        report["verdicts"] = {s["name"]: s.get("status") for s in report["sections"]}
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"D70_PROBE_DONE {report['status']} "
        f"{json.dumps(report.get('verdicts', {}))}",
        flush=True,
    )
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
