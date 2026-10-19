#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Attribute kernel-only duration through launch correlations, excluding annotations."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path


STAGES = ("zero_grad", "forward", "backward", "clip", "optimizer")


def analyze(events: list[dict]) -> dict:
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
    runtimes = {
        e["args"]["correlation"]: e
        for e in events
        if e.get("cat") == "cuda_runtime" and "correlation" in e.get("args", {})
    }
    operations = {
        e["args"]["External id"]: e
        for e in events
        if e.get("cat") == "cpu_op" and "External id" in e.get("args", {})
    }
    stages = [
        e
        for e in events
        if e.get("cat") == "user_annotation"
        and e.get("ph") == "X"
        and e.get("name") in {"n31/" + name for name in STAGES}
    ]
    grouped = defaultdict(lambda: dict(calls=0, gpu_us=0.0))
    by_stage = {name: dict(kernel_calls=0, gpu_us=0.0) for name in STAGES}
    by_operation = defaultdict(lambda: dict(kernel_calls=0, gpu_us=0.0))
    by_mm_shape = defaultdict(lambda: dict(kernel_calls=0, gpu_us=0.0))
    intervals = defaultdict(list)
    for kernel in kernels:
        duration = kernel["dur"]
        assert math.isfinite(duration) and duration > 0
        runtime = runtimes[kernel["args"]["correlation"]]
        # CPU submit timestamps classify autograd worker launches as well as
        # main-thread launches. GPU annotation ranges are not kernel records.
        owners = [e for e in stages if e["ts"] <= runtime["ts"] < e["ts"] + e["dur"]]
        assert len(owners) == 1, (kernel["name"], owners)
        stage = owners[0]["name"].partition("/")[2]
        by_stage[stage]["kernel_calls"] += 1
        by_stage[stage]["gpu_us"] += duration
        row = grouped[kernel["name"]]
        row["calls"] += 1
        row["gpu_us"] += duration
        operation = operations.get(kernel["args"].get("External id"), {})
        if operation.get("name") == "aten::mm":
            shape = json.dumps(
                dict(
                    input_shapes=operation.get("args", {}).get("Input Dims"),
                    input_types=operation.get("args", {}).get("Input type"),
                ),
                sort_keys=True,
            )
            row = by_mm_shape[shape]
            row["kernel_calls"] += 1
            row["gpu_us"] += duration
        row = by_operation[operation.get("name", "unmapped")]
        row["kernel_calls"] += 1
        row["gpu_us"] += duration
        intervals[kernel["args"].get("device", 0)].append(
            (kernel["ts"], kernel["ts"] + duration)
        )
    assert kernels
    union_us = 0.0
    for device_intervals in intervals.values():
        ordered = sorted(device_intervals)
        begin, end = ordered[0]
        for a, b in ordered[1:]:
            if a > end:
                union_us += end - begin
                begin, end = a, b
            else:
                end = max(end, b)
        union_us += end - begin
    total = sum(e["dur"] for e in kernels)
    return dict(
        kernel_count=len(kernels),
        kernel_sum_us=total,
        kernel_device_union_us=union_us,
        stages=by_stage,
        top_kernels=[
            dict(name=name, **row, fraction_of_kernel_sum=row["gpu_us"] / total)
            for name, row in sorted(
                grouped.items(), key=lambda p: p[1]["gpu_us"], reverse=True
            )[:15]
        ],
        top_launch_operations=[
            dict(name=name, **row, fraction_of_kernel_sum=row["gpu_us"] / total)
            for name, row in sorted(
                by_operation.items(), key=lambda p: p[1]["gpu_us"], reverse=True
            )[:15]
        ],
        top_mm_shapes=[
            dict(
                **json.loads(shape), **row, fraction_of_kernel_sum=row["gpu_us"] / total
            )
            for shape, row in sorted(
                by_mm_shape.items(), key=lambda p: p[1]["gpu_us"], reverse=True
            )[:15]
        ],
        excluded_gpu_annotations=sum(
            e.get("cat") == "gpu_user_annotation" for e in events
        ),
        method="kernel category only; launch correlation to CPU submit time partitions sequential training stages across CPU threads; External id maps launch operations; sums and per-device interval union reported separately",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(json.loads(args.trace.read_text())["traceEvents"])
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
