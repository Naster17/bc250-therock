#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""D70 summarizer (pure logic, workstation-testable).

Combines the feasibility probe verdicts, the instrumented LoRA decomposition
and the optional one-update kernel census into one decision summary: does the
instrumentation reproduce the D69 reference step, where does the step go, and
which launch-reduction lever is feasible?
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import decompose_lora_d70 as dec  # noqa: E402

D69_REF_SECONDS = 36.166
D69_4096_REF_SECONDS = 324.172
D69_REF_LOSS = 2.4717
REPRO_BOUND = 0.05
LOSS_BOUND = 1e-3


def cold_update_summary(decomp_4096: dict) -> dict:
    row = decomp_4096["updates"][0]
    ratio = row["wall_seconds"] / D69_4096_REF_SECONDS
    return {
        "wall_seconds": row["wall_seconds"],
        "d69_ref_seconds": D69_4096_REF_SECONDS,
        "ratio": ratio,
        "within_bound": abs(ratio - 1.0) <= REPRO_BOUND,
        "top": row["top"],
        "shares": row["shares"],
        "sub": row["sub"],
        "cpu": row.get("cpu"),
        "gpu_timeline_coverage": row.get("gpu_timeline_coverage"),
        "final_loss": decomp_4096.get("final_loss"),
    }


def build_summary(
    probe: dict,
    decomp: dict,
    profile: dict | None,
    decomp_4096: dict | None = None,
) -> dict:
    reproduction = dec.build_summary(decomp, D69_REF_SECONDS, REPRO_BOUND, LOSS_BOUND)
    verdicts = probe.get("verdicts", {})
    failed_sections = probe.get("failed_sections", [])
    shares = reproduction["steady_shares"]
    ranked = sorted(shares.items(), key=lambda kv: kv[1], reverse=True)
    reasons = []
    if not reproduction["reproduction_within_bound"]:
        reasons.append(
            f"instrumented step {reproduction['steady_wall_seconds']:.3f}s "
            f"vs D69 {D69_REF_SECONDS}s outside ±{REPRO_BOUND:.0%}"
        )
    if not reproduction["loss_within_bound"]:
        reasons.append(
            f"final loss delta {reproduction['final_loss_delta']} > {LOSS_BOUND}"
        )
    if verdicts.get("instrumentation_selftest") != "works":
        reasons.append("instrumentation selftest did not pass")
    if failed_sections:
        reasons.append(f"probe fixture failures: {failed_sections}")
    cold_4096 = None
    if decomp_4096 is not None and decomp_4096.get("updates"):
        cold_4096 = cold_update_summary(decomp_4096)
        if not cold_4096["within_bound"]:
            reasons.append(
                f"4096 cold update {cold_4096['wall_seconds']:.3f}s vs D69 "
                f"{D69_4096_REF_SECONDS}s outside ±{REPRO_BOUND:.0%}"
            )
    return {
        "status": "passed" if not reasons else "failed",
        "reasons": reasons,
        "reproduction": reproduction,
        "cold_4096": cold_4096,
        "probe_verdicts": verdicts,
        "graph_feasibility": {
            "mlp_capture": verdicts.get("graph_mlp"),
            "qwen35_aten_capture": verdicts.get("graph_qwen35_aten"),
            "qwen35_fused_triton_capture": verdicts.get("graph_qwen35_fused"),
            "checkpoint_capture": verdicts.get("graph_checkpoint"),
            "make_graphed_callables": verdicts.get("graphed_callables"),
        },
        "profiler_available": verdicts.get("profiler") == "works",
        "ranked_step_shares": [
            {"category": cat, "share": share} for cat, share in ranked
        ],
        "kernel_census": profile,
    }
