#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""D70 owned LoRA step decomposition wrapper.

Instruments the unchanged lora-smoke two-host entry point with transparent
CUDA-event spans and class-level phase patches, then forwards argv verbatim
(D69 wrapper pattern). No application source, entry point, output or checkpoint
is modified; the instrumentation is math-neutral (events and wall timestamps
only, no RNG consumption, no extra synchronization inside the step).

GPU timeline categories (disjoint event spans per optimizer update):
    mixer/full_attention, mixer/linear_attention  (layer span minus MLP span)
    mlp                                           (forward + backward spans)
    fwd_other, bwd_other                          (embed/norms/head/loss/LoRA glue)
    clip, optimizer                               (optimizer = prepared optimizer
                                                   step: FP16 scaler + AdamW)
CPU wall structure (overlaps GPU execution; reported separately):
    micro_wall, inter_micro_gap, pre_optimizer_gap, post_clip_gap, clip_wall,
    optimizer_wall, host_gap (sum of the gaps).
Sub-attribution (informational, nested): fn/conv, fn/chunk, fn/eager spans,
split by phase into original forward vs checkpoint recompute.

Span semantics under gradient checkpointing (established by diag1 on this
stack): module forward spans cover the ORIGINAL forward only — the recomputed
forward fires the pre hook but is interrupted before the post hook. Module
backward spans fire exactly once per microbatch; any recompute time they do
not cover lands in bwd_other, and the fn/* recompute spans quantify the
conv/chunk/eager part of it. The budget closes by construction.

Usage (inside the qualified worker, via the guardian):
    python3 decompose_lora_d70.py --config <app-config.json> [app args...]

Environment knobs:
    BC250_D70_DECOMPOSE=0   run the entry point without instrumentation
    BC250_D70_PROFILE=1     additionally torch.profiler one steady update
                            (starts after optimizer step 1, stops after step 2)

Writes decomposition-rank<N>.json (and optionally profile-rank<N>.json) into
the app output directory. Spans are finalized after the entry point returns,
so no extra torch.cuda.synchronize is inserted into the measured steps.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import runpy
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT = Path("/home/nik/bc250-projects/lora-smoke/project")
ENTRY = PROJECT / "train_two_host.py"
MODELING = "transformers.models.qwen3_5.modeling_qwen3_5"
FUNCTION_CATEGORIES = {
    "causal_conv1d_fn": "fn/conv",
    "torch_chunk_gated_delta_rule": "fn/chunk",
    "eager_attention_forward": "fn/eager",
}


# ---------------------------------------------------------------------------
# Pure logic (workstation-testable, no torch import).
# ---------------------------------------------------------------------------
def extract_config(argv: list[str]) -> Path:
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if arg.startswith("--config="):
            return Path(arg.split("=", 1)[1])
    raise ValueError("wrapper requires --config <app-config.json>")


def layer_category(block_type: str) -> str:
    if block_type not in ("linear_attention", "full_attention"):
        raise ValueError(f"unknown block_type {block_type!r}")
    return f"layer/{block_type}"


def _sum(spans, **filt) -> float:
    total = 0.0
    for s in spans:
        if all(s.get(k) == v for k, v in filt.items()):
            total += s["seconds"]
    return total


def aggregate_update(
    spans: list[dict], update: int, grad_accum: int, layer_types: list[str]
) -> dict:
    """Per-update attribution from resolved spans (seconds/wall floats).

    Two complementary views, because the pipeline is asynchronous:
    - GPU timeline: disjoint event spans (mixer types, mlp, other, clip,
      optimizer). Their sum / wall is the timeline coverage (~1 when the
      spans tile the update); spans include launch gaps, so this is where
      the GPU timeline goes, not pure kernel busy time.
    - CPU wall structure: microbatch call walls plus the gaps between them
      (dataloader/callbacks), before the optimizer (audit finite-check sync
      + GPU drain) and after clipping. These overlap GPU execution and must
      NOT be added to the GPU spans as if disjoint.
    """
    first = update * grad_accum
    micro_ids = set(range(first, first + grad_accum))
    sel = [s for s in spans if s["micro"] in micro_ids]
    phase_spans = [s for s in spans if s.get("update") == update]
    micro = [s for s in sel if s["cat"] == "micro_total"]
    bwd = [s for s in sel if s["cat"] == "backward_total"]
    clip = [s for s in phase_spans if s["cat"] == "clip"]
    opt = [s for s in phase_spans if s["cat"] == "optimizer"]
    if len(micro) != grad_accum or len(opt) != 1:
        raise ValueError(
            f"update {update}: {len(micro)} micro spans, {len(opt)} optimizer spans"
        )
    micro_wall = sum(s["wall"] for s in micro)
    micro_gpu = sum(s["seconds"] for s in micro)
    bwd_gpu = sum(s["seconds"] for s in bwd)
    fwd_gpu = micro_gpu - bwd_gpu

    def layer_type(span):
        idx = span.get("layer")
        return layer_types[idx] if idx is not None and idx < len(layer_types) else None

    mixer_fwd, mixer_bwd, mlp_fwd, mlp_bwd = {}, {}, {}, {}
    for bt in ("full_attention", "linear_attention"):
        lf = _sum(sel, cat=f"layer/{bt}", phase="forward")
        lb = _sum(sel, cat=f"layer/{bt}", phase="backward")
        mf = sum(
            s["seconds"]
            for s in sel
            if s["cat"] == "mod/mlp" and s["phase"] == "forward" and layer_type(s) == bt
        )
        mb = sum(
            s["seconds"]
            for s in sel
            if s["cat"] == "mod/mlp"
            and s["phase"] == "backward"
            and layer_type(s) == bt
        )
        mixer_fwd[bt], mixer_bwd[bt] = lf - mf, lb - mb
        mlp_fwd[bt], mlp_bwd[bt] = mf, mb
    fwd_other = fwd_gpu - sum(mixer_fwd.values()) - sum(mlp_fwd.values())
    bwd_other = bwd_gpu - sum(mixer_bwd.values()) - sum(mlp_bwd.values())

    sub = {}
    for cat in ("fn/conv", "fn/chunk", "fn/eager"):
        sub[cat] = {
            "forward": _sum(sel, cat=cat, phase="forward"),
            "recompute": _sum(sel, cat=cat, phase="backward"),
        }
    clip_wall = sum(s["wall"] for s in clip)
    opt_wall = sum(s["wall"] for s in opt)
    micro_sorted = sorted(micro, key=lambda s: s["w0"])
    update_wall = opt[0]["w1"] - micro_sorted[0]["w0"]
    inter_micro_gap = sum(
        b["w0"] - a["w1"] for a, b in zip(micro_sorted, micro_sorted[1:])
    )
    pre_optimizer_gap = (clip[0]["w0"] if clip else opt[0]["w0"]) - micro_sorted[-1][
        "w1"
    ]
    post_clip_gap = opt[0]["w0"] - clip[0]["w1"] if clip else 0.0
    host_gap = inter_micro_gap + pre_optimizer_gap + post_clip_gap
    top = {
        "mixer/full_attention": mixer_fwd["full_attention"]
        + mixer_bwd["full_attention"],
        "mixer/linear_attention": mixer_fwd["linear_attention"]
        + mixer_bwd["linear_attention"],
        "mlp": sum(mlp_fwd.values()) + sum(mlp_bwd.values()),
        "fwd_other": fwd_other,
        "bwd_other": bwd_other,
        "clip": sum(s["seconds"] for s in clip),
        "optimizer": sum(s["seconds"] for s in opt),
    }
    budget_gpu = sum(top.values())
    return {
        "update": update,
        "wall_seconds": update_wall,
        "micro_wall_seconds": micro_wall,
        "top": top,
        "shares": {k: v / update_wall for k, v in top.items()},
        "gpu_timeline_coverage": budget_gpu / update_wall,
        "cpu": {
            "micro_wall": micro_wall,
            "inter_micro_gap": inter_micro_gap,
            "pre_optimizer_gap": pre_optimizer_gap,
            "post_clip_gap": post_clip_gap,
            "clip_wall": clip_wall,
            "optimizer_wall": opt_wall,
            "host_gap": host_gap,
            "closure_residual": update_wall
            - (micro_wall + host_gap + clip_wall + opt_wall),
        },
        "sub": sub,
        "recompute_total": sum(v["recompute"] for v in sub.values()),
    }


def steady_median(rows: list[dict], key: str) -> float:
    """D67 protocol: exclude the first update, median of the rest."""
    values = sorted(r[key] for r in rows[1:])
    if not values:
        raise ValueError("no steady rows")
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def progress_label(updates: list[dict]) -> str:
    """Single-update runs (cold feasibility cases) have no steady rows."""
    if len(updates) > 1:
        return f"steady={steady_median(updates, 'wall_seconds'):.3f}s"
    return f"cold={updates[0]['wall_seconds']:.3f}s"


def build_summary(
    decomp: dict,
    d69_ref_seconds: float,
    repro_bound: float = 0.05,
    loss_bound: float = 1e-3,
) -> dict:
    rows = decomp["updates"]
    median = steady_median(rows, "wall_seconds")
    ratio = median / d69_ref_seconds
    final_loss = decomp.get("final_loss")
    loss_delta = (
        abs(final_loss - decomp["d69_ref_loss"])
        if final_loss is not None and decomp.get("d69_ref_loss") is not None
        else None
    )
    shares = {}
    for cat in rows[0]["top"]:
        shares[cat] = steady_median([{"s": r["shares"][cat]} for r in rows], "s")
    steady_cpu = None
    coverage = None
    if "cpu" in rows[0]:
        steady_cpu = {
            k: steady_median([{"v": r["cpu"][k]} for r in rows], "v")
            for k in rows[0]["cpu"]
        }
        coverage = steady_median([{"v": r["gpu_timeline_coverage"]} for r in rows], "v")
    return {
        "steady_wall_seconds": median,
        "d69_ref_seconds": d69_ref_seconds,
        "reproduction_ratio": ratio,
        "reproduction_within_bound": abs(ratio - 1.0) <= repro_bound,
        "final_loss": final_loss,
        "final_loss_delta": loss_delta,
        "loss_within_bound": loss_delta is None or loss_delta <= loss_bound,
        "steady_shares": shares,
        "steady_cpu": steady_cpu,
        "gpu_timeline_coverage": coverage,
        "steady_sub": {
            cat: {
                phase: steady_median([{"s": r["sub"][cat][phase]} for r in rows], "s")
                for phase in ("forward", "recompute")
            }
            for cat in ("fn/conv", "fn/chunk", "fn/eager")
        },
    }


# ---------------------------------------------------------------------------
# Torch-dependent instrumentation (board only).
# ---------------------------------------------------------------------------
class Recorder:
    def __init__(self, torch, grad_accum: int):
        self.torch = torch
        self.grad_accum = grad_accum
        self.spans = []
        self.phase = "idle"
        self.micro = -1
        self.update_completed = 0
        self.current_layer = None
        self.unpaired = 0
        self.fires = {}
        self.pending_stacks = []
        self.pending_leftover = 0

    def _event(self):
        return self.torch.cuda.Event(enable_timing=True)

    def _append(self, cat, start, end, w0, w1, phase, micro, layer, update):
        self.spans.append(
            {
                "cat": cat,
                "start": start,
                "end": end,
                "w0": w0,
                "w1": w1,
                "phase": phase,
                "micro": micro,
                "layer": layer,
                "update": update,
            }
        )

    @contextmanager
    def span(self, cat, update=None):
        start, end = self._event(), self._event()
        w0 = time.monotonic()
        start.record()
        try:
            yield
        finally:
            end.record()
            w1 = time.monotonic()
            self._append(
                cat,
                start,
                end,
                w0,
                w1,
                self.phase,
                self.micro,
                self.current_layer,
                self.update_completed if update is None else update,
            )

    def hook_pair(self, cat, layer_idx=None):
        """Forward or full-backward pre/post hook pair with its own pending
        stack.

        Checkpointing quirk (diag1, this stack): the recomputed forward fires
        the PRE hook but is interrupted before the POST hook, leaving one
        dangling pending entry per recompute. LIFO pairing stays correct for
        the original forwards (their pre is always on top when their post
        fires); leftovers are counted at finalize. Layer context is set, not
        push/popped, for the same reason."""
        pending = []
        rec = self
        rec.pending_stacks.append(pending)

        def pre(module, *args):
            rec.fires[cat + ":pre"] = rec.fires.get(cat + ":pre", 0) + 1
            ev = rec._event()
            ev.record()
            pending.append(
                (ev, time.monotonic(), rec.phase, rec.micro, rec.current_layer)
            )
            if layer_idx is not None:
                rec.current_layer = layer_idx

        def post(module, *args):
            rec.fires[cat + ":post"] = rec.fires.get(cat + ":post", 0) + 1
            if not pending:
                rec.unpaired += 1
                return
            ev, w0, phase, micro, layer = pending.pop()
            end = rec._event()
            end.record()
            rec._append(
                cat,
                ev,
                end,
                w0,
                time.monotonic(),
                phase,
                micro,
                layer,
                micro // rec.grad_accum if micro >= 0 else None,
            )

        return pre, post

    def finalize(self) -> list[dict]:
        self.torch.cuda.synchronize()
        self.pending_leftover = sum(len(p) for p in self.pending_stacks)
        resolved = []
        for s in self.spans:
            resolved.append(
                {
                    "cat": s["cat"],
                    "phase": s["phase"],
                    "micro": s["micro"],
                    "layer": s["layer"],
                    "update": s["update"],
                    "seconds": s["start"].elapsed_time(s["end"]) / 1000.0,
                    "wall": s["w1"] - s["w0"],
                    "w0": s["w0"],
                    "w1": s["w1"],
                }
            )
        self.spans = []
        return resolved


def install_instrumentation(model, recorder, handles):
    """Module hooks on the (PEFT-wrapped) model; handles collected for exact
    removal. Called lazily on the first training_step, when the model exists."""
    layer_types = []
    for name, module in model.named_modules():
        cls = type(module).__name__
        if cls == "Qwen3_5DecoderLayer":
            idx = len(layer_types)
            layer_types.append(module.block_type)
            cat = layer_category(module.block_type)
            pre, post = recorder.hook_pair(cat, layer_idx=idx)
            handles.append(module.register_forward_pre_hook(pre))
            handles.append(module.register_forward_hook(post))
            bpre, bpost = recorder.hook_pair(cat, layer_idx=idx)
            handles.append(module.register_full_backward_pre_hook(bpre))
            handles.append(module.register_full_backward_hook(bpost))
            mpre, mpost = recorder.hook_pair("mod/mlp", layer_idx=idx)
            handles.append(module.mlp.register_forward_pre_hook(mpre))
            handles.append(module.mlp.register_forward_hook(mpost))
            mbpre, mbpost = recorder.hook_pair("mod/mlp", layer_idx=idx)
            handles.append(module.mlp.register_full_backward_pre_hook(mbpre))
            handles.append(module.mlp.register_full_backward_hook(mbpost))
    if not layer_types:
        raise ValueError("no Qwen3_5DecoderLayer modules found")
    return layer_types


def wrap_functions(modeling, recorder, restores):
    """Transparent spans around the module-level reference functions. The
    eager interface may come from the attention registry instead of the module
    global; wrap both when they differ and count which one fires."""
    for fname, cat in FUNCTION_CATEGORIES.items():
        original = getattr(modeling, fname)

        def make(original=original, cat=cat):
            def wrapped(*args, **kwargs):
                with recorder.span(cat):
                    return original(*args, **kwargs)

            return wrapped

        wrapped = make()
        setattr(modeling, fname, wrapped)
        restores.append((modeling, fname, original, wrapped))
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        registry_fn = ALL_ATTENTION_FUNCTIONS.get_interface("eager", None)
        module_fn = modeling.eager_attention_forward
        if registry_fn is not None and registry_fn is not module_fn:

            def reg_wrapped(*args, **kwargs):
                with recorder.span("fn/eager"):
                    return registry_fn(*args, **kwargs)

            ALL_ATTENTION_FUNCTIONS["eager"] = reg_wrapped
            restores.append(
                (ALL_ATTENTION_FUNCTIONS, "eager", registry_fn, reg_wrapped)
            )
    except Exception as exc:  # registry shape differs; module wrap still active
        print(f"D70_REGISTRY wrap skipped: {exc!r}", flush=True)


def uninstall(restores, handles):
    for holder, name, original, wrapped in restores:
        if isinstance(holder, dict):
            current = holder.get(name)
        else:
            current = getattr(holder, name)
        if current is not wrapped:
            raise RuntimeError(
                f"foreign patch on {name}; refusing to restore: {current!r}"
            )
        if isinstance(holder, dict):
            holder[name] = original
        else:
            setattr(holder, name, original)
    for handle in handles:
        handle.remove()


def main() -> int:
    import torch

    config_path = extract_config(sys.argv[1:])
    app_config = json.loads(config_path.read_text())
    rank = int(os.environ.get("RANK", "0"))
    output_dir = Path(app_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    enabled = os.environ.get("BC250_D70_DECOMPOSE", "1") != "0"
    profile_enabled = os.environ.get("BC250_D70_PROFILE", "0") == "1"
    grad_accum = int(app_config["grad_accum"])
    summary = {
        "fixture": "decompose_lora_d70",
        "enabled": enabled,
        "profile_enabled": profile_enabled,
        "rank": rank,
        "config": str(config_path),
        "grad_accum": grad_accum,
    }
    if not enabled:
        sys.path.insert(0, str(PROJECT))
        sys.argv = [str(ENTRY), *sys.argv[1:]]
        runpy.run_path(str(ENTRY), run_name="__main__")
        return 0

    import transformers
    from accelerate import Accelerator
    import importlib

    modeling = importlib.import_module(MODELING)

    recorder = Recorder(torch, grad_accum)
    restores, handles = [], []
    state = {
        "installed": False,
        "layer_types": [],
        "prof": None,
        "prof_started": False,
        "prof_stopped": False,
        "opt": None,
        "opt_step": None,
    }
    wrap_functions(modeling, recorder, restores)

    orig_training_step = transformers.Trainer.training_step

    def training_step(self, model, inputs, num_items_in_batch=None):
        if not state["installed"]:
            state["layer_types"] = install_instrumentation(model, recorder, handles)
            # Instance-level patch on the prepared optimizer (the Trainer
            # calls self.optimizer.step()). accelerate re-patches the INNER
            # optimizer's instance step around every FP16 scaler step, so a
            # class-level AdamW patch is not stable here; the prepared
            # wrapper's own step attribute is untouched by accelerate.
            opt = self.optimizer
            orig_opt_step = opt.step

            def opt_step(closure=None):
                with recorder.span("optimizer", update=recorder.update_completed):
                    result = orig_opt_step(closure)
                recorder.update_completed += 1
                if profile_enabled:
                    from torch.profiler import profile, ProfilerActivity

                    if recorder.update_completed == 1 and not state["prof_started"]:
                        state["prof"] = profile(
                            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
                        )
                        state["prof"].start()
                        state["prof_started"] = True
                        print("D70_PROFILE started after update 1", flush=True)
                    elif recorder.update_completed == 2 and state["prof"] is not None:
                        state["prof"].stop()
                        state["prof_stopped"] = True
                        print("D70_PROFILE stopped after update 2", flush=True)
                return result

            opt.step = opt_step
            state["opt"] = opt
            state["opt_step"] = opt_step
            state["installed"] = True
            print(
                f"D70_HOOKS installed layers={len(state['layer_types'])} "
                f"types={state['layer_types']}",
                flush=True,
            )
        recorder.micro += 1
        recorder.phase = "forward"
        with recorder.span("micro_total"):
            try:
                return orig_training_step(self, model, inputs, num_items_in_batch)
            finally:
                recorder.phase = "idle"

    transformers.Trainer.training_step = training_step
    restores.append(
        (transformers.Trainer, "training_step", orig_training_step, training_step)
    )

    orig_backward = Accelerator.backward

    def backward(self, loss, **kwargs):
        recorder.phase = "backward"
        with recorder.span("backward_total"):
            try:
                return orig_backward(self, loss, **kwargs)
            finally:
                recorder.phase = "idle"

    Accelerator.backward = backward
    restores.append((Accelerator, "backward", orig_backward, backward))

    if hasattr(Accelerator, "clip_grad_norm_"):
        orig_clip = Accelerator.clip_grad_norm_

        def clip(self, *args, **kwargs):
            with recorder.span("clip"):
                return orig_clip(self, *args, **kwargs)

        Accelerator.clip_grad_norm_ = clip
        restores.append((Accelerator, "clip_grad_norm_", orig_clip, clip))

    exit_code = 0
    try:
        sys.path.insert(0, str(PROJECT))
        sys.argv = [str(ENTRY), *sys.argv[1:]]
        runpy.run_path(str(ENTRY), run_name="__main__")
    except SystemExit as exc:
        exit_code = exc.code or 0
        raise
    finally:
        try:
            if state.get("opt") is not None:
                if state["opt"].step is not state["opt_step"]:
                    raise RuntimeError(
                        "foreign patch on optimizer.step; refusing to restore: "
                        f"{state['opt'].step!r}"
                    )
                del state["opt"].step
            uninstall(restores, handles)
            summary["uninstalled"] = True
            resolved = recorder.finalize()
            updates = [
                aggregate_update(resolved, u, grad_accum, state["layer_types"])
                for u in range(recorder.update_completed)
            ]
            audit_path = output_dir / f"audit-rank{rank}.json"
            audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
            final_loss = None
            if audit and audit.get("rows"):
                metrics = audit["rows"][-1].get("metrics", {})
                final_loss = metrics.get("loss")
            summary.update(
                {
                    "versions": {
                        "torch": torch.__version__,
                        "transformers": transformers.__version__,
                    },
                    "layer_types": state["layer_types"],
                    "span_count": len(resolved),
                    "unpaired_hooks": recorder.unpaired,
                    "hook_fires": recorder.fires,
                    "pending_leftover": recorder.pending_leftover,
                    "updates": updates,
                    "spans": resolved,
                    "audit_seconds": (
                        [r["seconds"] for r in audit["rows"]] if audit else None
                    ),
                    "final_loss": final_loss,
                    "d69_ref_loss": 2.4717,
                    "coverage": {
                        cat: sum(1 for s in resolved if s["cat"] == cat)
                        for cat in sorted({s["cat"] for s in resolved})
                    },
                }
            )
            if state["prof"] is not None and state["prof_started"]:
                if not state["prof_stopped"]:
                    state["prof"].stop()
                    state["prof_stopped"] = True
                from torch.autograd import DeviceType

                ka = state["prof"].key_averages()

                def entry(e):
                    return {
                        "name": e.key,
                        "count": e.count,
                        "self_device_ms": e.self_device_time_total / 1000.0,
                    }

                # Kernel events and their parent aten:: ops BOTH carry device
                # time; summing both double-counts. Separate by device_type.
                kernels = [entry(e) for e in ka if e.device_type == DeviceType.CUDA]
                ops = [
                    entry(e)
                    for e in ka
                    if e.device_type == DeviceType.CPU and e.self_device_time_total > 0
                ]
                kernels.sort(key=lambda k: k["self_device_ms"], reverse=True)
                ops.sort(key=lambda k: k["self_device_ms"], reverse=True)
                profile_report = {
                    "kernel_launches": sum(k["count"] for k in kernels),
                    "distinct_kernels": len(kernels),
                    "device_busy_ms": sum(k["self_device_ms"] for k in kernels),
                    "op_device_ms_double_counted": sum(
                        k["self_device_ms"] for k in ops
                    ),
                    "top_kernels": kernels[:40],
                    "top_ops": ops[:15],
                }
                (output_dir / f"profile-rank{rank}.json").write_text(
                    json.dumps(profile_report, indent=2) + "\n"
                )
                summary["profile"] = {
                    "kernel_launches": profile_report["kernel_launches"],
                    "distinct_kernels": profile_report["distinct_kernels"],
                }
            (output_dir / f"decomposition-rank{rank}.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
            if updates:
                print(
                    f"D70_DECOMPOSE updates={len(updates)} "
                    f"{progress_label(updates)} "
                    f"spans={len(resolved)} unpaired={recorder.unpaired}",
                    flush=True,
                )
        except BaseException as exc:
            summary["finalize_error"] = repr(exc)
            (output_dir / f"decomposition-rank{rank}.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
            raise
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
