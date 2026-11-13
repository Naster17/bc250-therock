#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""D71 per-shape in-model GEMM profiling.

Runs the unchanged app entry for 2+ updates under torch.profiler with
record_shapes, profiles exactly the second (steady) update — started after
optimizer step 1, stopped after optimizer step 2 — and dumps the
mm/bmm/addmm key_averages grouped by input shape with counts and self device
time. This resolves the D71 fixture-vs-model gap: the private-handle
fixtures project ~12-13 s of GEMM per update while D70's kernel census
attributed ~32 s in-model. Per-shape in-model times show which layouts (if
any) are actually slow inside the step, and whether the kernel-level
attribution is consistent with the op-level one.

The optimizer hook is an INSTANCE patch on the prepared optimizer (D70
lesson: accelerate re-patches the inner optimizer around every FP16 scaler
step, so class-level AdamW patches are not stable here).

Usage (guardian-bounded, app environment):
    python3 profile_shapes_d71.py --config <app-config.json> [app args...]
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import sys

PROJECT = Path("/home/nik/bc250-projects/lora-smoke/project")
ENTRY = PROJECT / "train_two_host.py"
GEMM_OPS = ("aten::mm", "aten::bmm", "aten::addmm")


def extract_config(argv: list[str]) -> Path:
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if arg.startswith("--config="):
            return Path(arg.split("=", 1)[1])
    raise ValueError("wrapper requires --config <app-config.json>")


def main() -> int:
    import torch
    from torch.profiler import ProfilerActivity, profile
    import transformers

    config_path = extract_config(sys.argv[1:])
    app_config = json.loads(config_path.read_text())
    rank = int(os.environ.get("RANK", "0"))
    output_dir = Path(app_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "prof": None,
        "opt_steps": 0,
        "stopped": False,
        "opt": None,
        "opt_step": None,
        "window_micros": 0,
    }
    # Profile exactly ONE microbatch of update 2 (a quarter of the events of
    # a full update). The full-update record_shapes profile OOM-killed the
    # 10g container and thrashed the host (r5/r6); a one-microbatch window
    # stays well inside the envelope. Per-update figures are x grad_accum.
    window_micros = 1
    orig_training_step = transformers.Trainer.training_step

    def training_step(self, model, inputs, num_items_in_batch=None):
        if state["opt"] is None:
            opt = self.optimizer
            orig_opt_step = opt.step

            def opt_step(closure=None):
                result = orig_opt_step(closure)
                state["opt_steps"] += 1
                if state["opt_steps"] == 1 and state["prof"] is None:
                    state["prof"] = profile(
                        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        record_shapes=True,
                    )
                    state["prof"].start()
                    print("D71_PROFILE_SHAPES started after update 1", flush=True)
                elif state["opt_steps"] == 2 and not state["stopped"]:
                    state["prof"].stop()
                    state["stopped"] = True
                    print("D71_PROFILE_SHAPES backstop stop after update 2", flush=True)
                return result

            opt.step = opt_step
            state["opt"] = opt
            state["opt_step"] = opt_step
        result = orig_training_step(self, model, inputs, num_items_in_batch)
        if state["prof"] is not None and not state["stopped"]:
            state["window_micros"] += 1
            if state["window_micros"] >= window_micros:
                state["prof"].stop()
                state["stopped"] = True
                print(
                    f"D71_PROFILE_SHAPES stopped after {state['window_micros']} "
                    "microbatch(es) of update 2",
                    flush=True,
                )
        return result

    transformers.Trainer.training_step = training_step

    exit_code = 0
    try:
        sys.path.insert(0, str(PROJECT))
        sys.argv = [str(ENTRY), *sys.argv[1:]]
        runpy.run_path(str(ENTRY), run_name="__main__")
    except SystemExit as exc:
        exit_code = exc.code or 0
    finally:
        transformers.Trainer.training_step = orig_training_step
        if state["opt"] is not None:
            if state["opt"].step is not state["opt_step"]:
                raise RuntimeError("foreign patch on optimizer.step")
            del state["opt"].step
        if state["prof"] is not None:
            if not state["stopped"]:
                state["prof"].stop()
            # Stream-aggregate events: key_averages(group_by_input_shape=True)
            # OOM-killed the 10g container on ~300k shape-tagged events (r5).
            gemm: dict = {}
            kernel_ms = 0.0
            kernel_rows: dict = {}
            for e in state["prof"].events():
                if e.device_type == torch.autograd.DeviceType.CUDA:
                    kernel_ms += e.self_device_time_total / 1000.0
                    k = kernel_rows.setdefault(
                        e.name, {"name": e.name, "count": 0, "self_device_ms": 0.0}
                    )
                    k["count"] += 1
                    k["self_device_ms"] += e.self_device_time_total / 1000.0
                if e.name in GEMM_OPS:
                    key = (e.name, str(e.input_shapes))
                    row = gemm.setdefault(
                        key,
                        {
                            "op": e.name,
                            "input_shapes": e.input_shapes,
                            "count": 0,
                            "self_device_ms": 0.0,
                        },
                    )
                    row["count"] += 1
                    row["self_device_ms"] += e.self_device_time_total / 1000.0
            rows = sorted(
                gemm.values(), key=lambda r: r["self_device_ms"], reverse=True
            )
            for r in rows:
                r["device_ms_per_call"] = (
                    r["self_device_ms"] / r["count"] if r["count"] else 0.0
                )
            kernels = sorted(
                kernel_rows.values(), key=lambda k: k["self_device_ms"], reverse=True
            )
            report = {
                "fixture": "profile_shapes_d71",
                "rank": rank,
                "window_microbatches": state["window_micros"],
                "grad_accum": int(app_config["grad_accum"]),
                "gemm_ops": rows,
                "gemm_self_device_ms_total": sum(r["self_device_ms"] for r in rows),
                "gemm_self_device_ms_per_update_est": sum(
                    r["self_device_ms"] for r in rows
                )
                * int(app_config["grad_accum"]),
                "kernel_self_device_ms_total": kernel_ms,
                "top_kernels": kernels[:20],
            }
            (output_dir / f"profile-shapes-rank{rank}.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(
                f"D71_PROFILE_SHAPES gemm_ms={report['gemm_self_device_ms_total']:.0f} "
                f"kernel_ms={report['kernel_self_device_ms_total']:.0f} "
                f"rows={len(rows)}",
                flush=True,
            )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
