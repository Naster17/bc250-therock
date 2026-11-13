#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""D69 summary logic (pure, workstation-testable).

Extracts steady-state (1024: first update excluded) and cold first-update
(4096/8192) statistics from the qualified lora-smoke benchmark audits, merges
the route records, and computes fused-vs-reference ratios. Refuses skipped or
non-finite updates and token-count disagreement, exactly like D67.
"""

from __future__ import annotations

import statistics


def steady_state(rows: list[dict]) -> dict:
    measured = rows[1:]
    if not measured:
        raise ValueError("no measured updates after warmup exclusion")
    for row in measured:
        if row["optimizer_skipped"] or not row["finite_gradients"]:
            raise ValueError(f"skipped/non-finite update at step {row['step']}")
    tokens = {row["input_tokens"] for row in measured}
    if len(tokens) != 1:
        raise ValueError(f"token counts disagree: {tokens}")
    seconds = [row["seconds"] for row in measured]
    med = statistics.median(seconds)
    ordered = sorted(seconds)
    n = len(ordered)
    q1 = statistics.median(ordered[: n // 2]) if n > 1 else med
    q3 = statistics.median(ordered[(n + 1) // 2 :]) if n > 1 else med
    return {
        "updates_run": len(rows),
        "updates_measured": len(measured),
        "median_s": med,
        "iqr_percent": (q3 - q1) / med * 100.0 if med > 0 else 0.0,
        "tokens_per_update": tokens.pop(),
    }


def cold_first(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("no updates")
    row = rows[0]
    if row["optimizer_skipped"] or not row["finite_gradients"]:
        raise ValueError(f"skipped/non-finite first update at step {row['step']}")
    return {
        "updates_run": len(rows),
        "first_update_s": row["seconds"],
        "tokens_per_update": row["input_tokens"],
        "tokens_per_second": row["input_tokens"] / row["seconds"],
        "peak_allocated_bytes": row["peak_allocated_bytes"],
    }


def summarize_case(audit: dict, route_record: dict | None, mode: str) -> dict:
    rows = audit["rows"]
    result = {
        "rank": audit["rank"],
        "world_size": audit["world_size"],
        "trainable_parameters": audit["trainable_parameters"],
        "mode": mode,
    }
    if mode == "steady":
        result.update(steady_state(rows))
        result["tokens_per_second"] = result["tokens_per_update"] / result["median_s"]
    else:
        result.update(cold_first(rows))
    if route_record is not None:
        result["route"] = {
            "target": route_record.get("target"),
            "route_enabled": route_record.get("route_enabled"),
            "native_calls": route_record.get("native_calls"),
            "fallbacks": route_record.get("fallbacks"),
            "uninstalled": route_record.get("uninstalled"),
        }
    return result


def ratio(before_s: float, after_s: float) -> float:
    return before_s / after_s if after_s > 0 else 0.0


def build_summary(cases: dict) -> dict:
    """cases: {name: summarize_case result}. Adds fused/ref ratios per length.

    Stopped cases (no timing keys) are skipped for ratios.
    """
    summary = {"cases": cases, "ratios": {}}
    for length in (1024, 4096, 8192):
        ref = cases.get(f"{length}-ref")
        fused = cases.get(f"{length}-fused")
        installed = cases.get(f"{length}-installed")

        def timed(case):
            return case is not None and ("median_s" in case or "first_update_s" in case)

        key = "median_s" if timed(ref) and "median_s" in ref else "first_update_s"
        if timed(ref) and timed(fused):
            summary["ratios"][f"{length}-fused-vs-ref"] = ratio(ref[key], fused[key])
        if timed(ref) and timed(installed):
            summary["ratios"][f"{length}-installed-vs-ref"] = ratio(
                ref[key], installed[key]
            )
    return summary
