#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""D71 summarizer (pure logic, workstation-testable).

Combines the GEMM census, the solution bake-off and the whole-step
reference/candidate A/B into one decision summary: does the reference
reproduce D69, does the candidate preset keep the loss trajectory within the
pre-declared bound, and what step-level gain did the qualified winners buy?
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preset_d71 as preset  # noqa: E402

D69_REF_SECONDS = 36.166
D69_REF_LOSS = 2.4717
REPRO_BOUND = 0.05
LOSS_BOUND = 0.01


def steady(rows: list[dict]) -> float:
    values = sorted(r["seconds"] for r in rows[1:])
    if not values:
        raise ValueError("no steady rows")
    return statistics.median(values)


def predicted_savings_seconds(
    bench: dict, min_ratio: float = preset.MIN_RATIO
) -> float:
    """Isolated-fixture savings projected onto one update's call counts.
    Indicative only — the whole-step A/B is the authoritative number."""
    total = 0.0
    for result in bench.get("results", []):
        if result["paired_gpu_ratio"] < min_ratio:
            continue
        saved_ms = result["medians"]["default"] - result["medians"]["candidate"]
        total += saved_ms * result["count"] / 1000.0
    return total


def build_summary(
    census: dict,
    bench: dict,
    ref_audit: dict,
    cand_audit: dict | None,
    preset_rows: int,
) -> dict:
    reasons = []
    ref_steady = steady(ref_audit["rows"])
    ratio = ref_steady / D69_REF_SECONDS
    if abs(ratio - 1.0) > REPRO_BOUND:
        reasons.append(
            f"reference step {ref_steady:.3f}s vs D69 {D69_REF_SECONDS}s "
            f"outside ±{REPRO_BOUND:.0%}"
        )
    ref_loss = ref_audit["rows"][-1]["metrics"].get("loss")
    cand = None
    if cand_audit is not None:
        cand_steady = steady(cand_audit["rows"])
        cand_loss = cand_audit["rows"][-1]["metrics"].get("loss")
        loss_delta = (
            abs(cand_loss - ref_loss)
            if cand_loss is not None and ref_loss is not None
            else None
        )
        if loss_delta is None or loss_delta > LOSS_BOUND:
            reasons.append(f"candidate final loss delta {loss_delta} > {LOSS_BOUND}")
        if not all(r["finite_gradients"] for r in cand_audit["rows"]):
            reasons.append("candidate non-finite gradients")
        if any(r["optimizer_skipped"] for r in cand_audit["rows"]):
            reasons.append("candidate skipped optimizer steps")
        cand = {
            "steady_seconds": cand_steady,
            "reference_steady_seconds": ref_steady,
            "step_speedup": ref_steady / cand_steady,
            "final_loss": cand_loss,
            "reference_final_loss": ref_loss,
            "final_loss_delta": loss_delta,
            "peak_gib": cand_audit["rows"][-1]["peak_allocated_bytes"] / 2**30,
        }
    winners = preset.winner_rows(bench)
    return {
        "status": "passed" if not reasons else "failed",
        "reasons": reasons,
        "census": {
            "total_gemm_calls": census["total_gemm_calls"],
            "distinct_layouts": census["distinct_layouts"],
        },
        "bench_status": bench.get("status"),
        "winners": winners,
        "preset_rows_added": preset_rows,
        "predicted_savings_seconds": predicted_savings_seconds(bench),
        "hipblaslt": bench.get("hipblaslt", []),
        "reference": {
            "steady_seconds": ref_steady,
            "d69_ref_seconds": D69_REF_SECONDS,
            "reproduction_ratio": ratio,
            "final_loss": ref_loss,
        },
        "candidate": cand,
    }
