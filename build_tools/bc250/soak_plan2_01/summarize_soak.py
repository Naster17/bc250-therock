#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PLAN2-01 pure summarizer: aggregate a chained-segment stability soak.

The soak accumulates a multi-hour LoRA run as many bounded guardian segments
(each <=460 s) joined by exact checkpoint/resume. This summarizer flattens the
per-segment audit rows into one continuous run and checks the pre-declared
reliability bounds: every update finite + not skipped, finite loss, contiguous
global steps (proves exact resume), adapter changed each segment, accumulated
training time >= target, no runaway per-update slowdown (drift), no cross-segment
memory growth (leak), and temperature within the envelope. No I/O; unit-testable.
"""

from __future__ import annotations

import statistics

# Pre-declared bounds (before any run).
DRIFT_MIN = 0.8
DRIFT_MAX = 1.3
MEMORY_RATIO_MAX = 1.15
TEMP_MAX_C = 90.0


def aggregate_rows(segment_audits: list[dict]) -> list[dict]:
    rows = []
    for audit in segment_audits:
        rows.extend(audit.get("rows", []))
    rows.sort(key=lambda r: r["step"])
    return rows


def training_seconds(rows: list[dict]) -> float:
    return sum(float(r["seconds"]) for r in rows)


def steps_contiguous(rows: list[dict]) -> bool:
    steps = [r["step"] for r in rows]
    return steps == list(range(1, len(rows) + 1))


def all_finite_nonskipped(rows: list[dict]) -> bool:
    return all(
        r.get("finite_gradients") and not r.get("optimizer_skipped") for r in rows
    )


def loss_finite(rows: list[dict]) -> bool:
    import math

    for r in rows:
        loss = r.get("metrics", {}).get("loss")
        if loss is None or not math.isfinite(loss):
            return False
    return True


def drift_ratio(rows: list[dict]) -> float:
    """Last-quartile vs first-quartile median per-update seconds (slowdown probe)."""
    seconds = [float(r["seconds"]) for r in rows]
    if len(seconds) < 4:
        return 1.0
    quarter = max(1, len(seconds) // 4)
    head = statistics.median(seconds[:quarter])
    tail = statistics.median(seconds[-quarter:])
    return tail / head if head > 0 else float("inf")


def adapters_changed(segment_audits: list[dict]) -> bool:
    return all(
        a.get("initial_adapter_sha256") != a.get("final_adapter_sha256")
        for a in segment_audits
    )


def segment_peaks(segment_audits: list[dict]) -> list[float]:
    peaks = []
    for audit in segment_audits:
        rows = audit.get("rows", [])
        if rows:
            peaks.append(max(float(r["peak_allocated_bytes"]) for r in rows))
    return peaks


def memory_ratio(segment_audits: list[dict]) -> float:
    peaks = segment_peaks(segment_audits)
    if not peaks:
        return 1.0
    med = statistics.median(peaks)
    return max(peaks) / med if med > 0 else float("inf")


def build_summary(
    segment_audits: list[dict],
    telemetry: list[dict],
    target_seconds: float,
    guardian_status: dict | None = None,
) -> dict:
    rows = aggregate_rows(segment_audits)
    if not rows:
        return dict(status="failed", error="no audit rows")
    temps = [t["temp_c"] for t in telemetry if t.get("temp_c") is not None]
    max_temp = max(temps) if temps else None
    checks = {
        "all_finite_nonskipped": all_finite_nonskipped(rows),
        "loss_finite": loss_finite(rows),
        "steps_contiguous": steps_contiguous(rows),
        "adapters_changed": adapters_changed(segment_audits),
        "accumulated_target_met": training_seconds(rows) >= target_seconds,
        "drift_within_bounds": DRIFT_MIN <= drift_ratio(rows) <= DRIFT_MAX,
        "memory_within_bounds": memory_ratio(segment_audits) <= MEMORY_RATIO_MAX,
        "temp_within_bounds": (max_temp is None) or (max_temp <= TEMP_MAX_C),
    }
    if guardian_status:
        checks["all_guardians_passed"] = all(guardian_status.values())
    losses = [r.get("metrics", {}).get("loss") for r in rows]
    losses = [x for x in losses if x is not None]
    summary = dict(
        status="passed" if all(checks.values()) else "failed",
        segments=len(segment_audits),
        total_updates=len(rows),
        accumulated_training_seconds=round(training_seconds(rows), 3),
        target_seconds=target_seconds,
        accumulated_hours=round(training_seconds(rows) / 3600.0, 3),
        mean_update_seconds=round(
            statistics.mean(float(r["seconds"]) for r in rows), 3
        ),
        drift_ratio=round(drift_ratio(rows), 4),
        memory_ratio=round(memory_ratio(segment_audits), 4),
        segment_peak_gib=[round(p / 2**30, 3) for p in segment_peaks(segment_audits)],
        max_temp_c=max_temp,
        first_loss=losses[0] if losses else None,
        last_loss=losses[-1] if losses else None,
        checks=checks,
        bounds={
            "drift": [DRIFT_MIN, DRIFT_MAX],
            "memory_ratio_max": MEMORY_RATIO_MAX,
            "temp_max_c": TEMP_MAX_C,
        },
    )
    if guardian_status:
        summary["guardian_status"] = guardian_status
    return summary
