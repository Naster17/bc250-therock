#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""D67 steady-state analysis of the owned LoRA benchmark audits.

Parses audit-rank*.json produced by the qualified L01 Audit callback under
BC250_LORA_BENCHMARK=1: excludes the first update (allocator/kernel warm-up),
requires every measured update finite and not optimizer-skipped, and summarizes
per-update seconds by median/IQR with global token counts and peak memory.
Throughput ratio = single-host median / two-host median (>1 means two boards
faster). Pure logic is unit-testable on the workstation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

FIXTURE = "summarize_d67"


def steady_state(rows: list[dict], exclude_first: int = 1) -> dict:
    measured = rows[exclude_first:]
    if not measured:
        raise ValueError("no measured updates after exclusion")
    if any(not r["finite_gradients"] or r["optimizer_skipped"] for r in measured):
        raise ValueError("measured update was nonfinite or optimizer-skipped")
    seconds = [r["seconds"] for r in measured]
    tokens = [r["input_tokens"] for r in measured]
    if len(set(tokens)) != 1:
        raise ValueError(f"unequal per-update token counts: {tokens}")
    median = statistics.median(seconds)
    return {
        "measured_updates": len(measured),
        "excluded_first": exclude_first,
        "step_median_s": median,
        "step_iqr_percent": (
            (
                statistics.quantiles(seconds, n=4)[2]
                - statistics.quantiles(seconds, n=4)[0]
            )
            / median
            * 100.0
            if len(seconds) >= 4
            else 0.0
        ),
        "step_samples_s": seconds,
        "global_tokens_per_update": tokens[0],
        "tokens_per_second": tokens[0] / median,
        "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in measured),
        "peak_reserved_bytes": max(r["peak_reserved_bytes"] for r in measured),
    }


def throughput_ratio(single: dict, ddp: dict) -> float:
    return single["step_median_s"] / ddp["step_median_s"]


def merge_ranks(audits: list[dict]) -> dict:
    """Rank audits carry identical global token counts; keep the max rank time."""
    if not audits:
        raise ValueError("no rank audits")
    worlds = {a["world_size"] for a in audits}
    if len(worlds) != 1:
        raise ValueError(f"mixed world sizes: {worlds}")
    rows = audits[0]["rows"]
    if len(audits) > 1:
        merged = []
        for i, row in enumerate(rows):
            other = audits[1]["rows"][i]
            if row["input_tokens"] != other["input_tokens"]:
                raise ValueError("rank token totals disagree")
            merged.append(
                {
                    **row,
                    "seconds": max(row["seconds"], other["seconds"]),
                    "peak_allocated_bytes": max(
                        row["peak_allocated_bytes"], other["peak_allocated_bytes"]
                    ),
                    "peak_reserved_bytes": max(
                        row["peak_reserved_bytes"], other["peak_reserved_bytes"]
                    ),
                }
            )
        rows = merged
    return {"world_size": audits[0]["world_size"], "rows": rows}


def summarize_case(case_dir: Path) -> dict:
    audits = [
        json.loads(p.read_text()) for p in sorted(case_dir.glob("audit-rank*.json"))
    ]
    merged = merge_ranks(audits)
    rows = merged["rows"]
    result = {
        "world_size": merged["world_size"],
        "trainable_parameters": audits[0]["trainable_parameters"],
        "updates_run": len(rows),
        "cold_first_update_s": rows[0]["seconds"],
        "cold_tokens_per_update": rows[0]["input_tokens"],
    }
    if len(rows) > 1:
        result.update(steady_state(rows))
    return result


def build_summary(evidence: Path) -> dict:
    cases = {}
    for case_dir in sorted(evidence.iterdir()):
        if not case_dir.is_dir() or not list(case_dir.glob("audit-rank*.json")):
            continue
        length, mode = case_dir.name.rsplit("-", 1)
        cases.setdefault(length, {})[mode] = summarize_case(case_dir)
    table = []
    for length, modes in sorted(cases.items(), key=lambda kv: int(kv[0])):
        entry = {"sequence_length": int(length), "modes": modes}
        if "single" in modes and "ddp" in modes:
            single, ddp = modes["single"], modes["ddp"]
            if "step_median_s" in single and "step_median_s" in ddp:
                entry["throughput_ratio_ddp_over_single"] = throughput_ratio(
                    single, ddp
                )
            entry["cold_first_update_ratio_ddp_over_single"] = (
                single["cold_first_update_s"] / ddp["cold_first_update_s"]
            )
        table.append(entry)
    return {"fixture": FIXTURE, "cases": table}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = build_summary(args.evidence)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    for entry in summary["cases"]:
        line = f"D67_SUMMARY seq={entry['sequence_length']}"
        for mode, data in sorted(entry["modes"].items()):
            if "step_median_s" in data:
                line += (
                    f" {mode}: {data['step_median_s']:.3f}s/update"
                    f" {data['tokens_per_second']:.0f}tok/s"
                )
            else:
                line += f" {mode}: cold={data['cold_first_update_s']:.3f}s"
        if "throughput_ratio_ddp_over_single" in entry:
            line += f" ratio={entry['throughput_ratio_ddp_over_single']:.3f}x"
        if "cold_first_update_ratio_ddp_over_single" in entry:
            line += (
                f" cold_ratio={entry['cold_first_update_ratio_ddp_over_single']:.3f}x"
            )
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
