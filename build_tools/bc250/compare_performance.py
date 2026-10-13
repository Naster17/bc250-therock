# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compare passed benchmark reports only when tensor bytes and controls match."""

import argparse
import json
from pathlib import Path


def compare(new: dict, old: dict) -> dict:
    if new["status"] != "passed" or old["status"] != "passed":
        raise ValueError("Both benchmark suites must pass correctness checks")
    for name in (
        "fixture_sha256",
        "arch",
        "kernel",
        "wgp_count",
        "cpu_affinity",
        "torch_threads",
        "sdma",
        "architecture_override",
        "samples",
        "sample_target_ms",
        "warmup_ms",
    ):
        if new[name] != old[name]:
            raise ValueError(f"Benchmark inputs/settings differ: {name}")
    old_cases = {case["name"]: case for case in old["cases"]}
    if set(old_cases) != {case["name"] for case in new["cases"]}:
        raise ValueError("Benchmark case sets differ")
    rows = []
    for current in new["cases"]:
        previous = old_cases[current["name"]]
        if current["status"] != "passed" or previous["status"] != "passed":
            raise ValueError(f"Case correctness did not pass: {current['name']}")
        if current["inputs_sha256"] != previous["inputs_sha256"]:
            raise ValueError(f"Tensor bytes differ: {current['name']}")
        if current["wall_median_ms"] <= 0 or previous["wall_median_ms"] <= 0:
            raise ValueError("Invalid timing")
        ratio = previous["wall_median_ms"] / current["wall_median_ms"]
        rows.append(
            {
                "name": current["name"],
                "new_ms": current["wall_median_ms"],
                "old_ms": previous["wall_median_ms"],
                "new_speedup": ratio,
                "new_time_change_percent": (
                    current["wall_median_ms"] / previous["wall_median_ms"] - 1
                )
                * 100,
                "new_iqr_percent": current["wall_iqr_percent"],
                "old_iqr_percent": previous["wall_iqr_percent"],
                "variable": max(
                    current["wall_iqr_percent"], previous["wall_iqr_percent"]
                )
                > 5,
                "new_first_ms": current["first_invocation_ms"],
                "old_first_ms": previous["first_invocation_ms"],
                "new_temperature_peak": current["peak_temperature_millicelsius"],
                "old_temperature_peak": previous["peak_temperature_millicelsius"],
            }
        )
    return {
        "status": "passed",
        "comparison": "deployed stacks on two boards; not a clock-normalized ROCm-only experiment",
        "clock_sensor_reliable": False,
        "fixture_sha256": new["fixture_sha256"],
        "new_torch": new["torch"],
        "old_torch": old["torch"],
        "new_hip": new["hip"],
        "old_hip": old["hip"],
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(json.loads(args.new.read_text()), json.loads(args.old.read_text()))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for row in report["cases"]:
        print(
            f"{row['name']}: new={row['new_ms']:.4f}ms old={row['old_ms']:.4f}ms speedup={row['new_speedup']:.3f}x variable={row['variable']}"
        )


if __name__ == "__main__":
    main()
