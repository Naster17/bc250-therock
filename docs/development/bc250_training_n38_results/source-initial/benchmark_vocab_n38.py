#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Full vocabulary GEMM scalar-solution audit with bounded FP64 reference tiles."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import time

import torch
from benchmark_gemm_n32 import Gemm, measure


def reference(a, b, role, column_tile=256, k_tile=2048):
    m, k = a.shape
    n = b.shape[1]
    assert b.shape[0] == k and column_tile > 0 and k_tile > 0
    out = torch.empty((m, n), dtype=torch.float64)
    started = time.monotonic()
    if role == "forward":
        left = a.double()
        for start in range(0, n, column_tile):
            out[:, start : start + column_tile] = (
                left @ b[:, start : start + column_tile].double()
            )
        scheme = "FP64 full-K,256 vocabulary columns per tile"
        owned_double_tensor_bound = (
            out.numel() * 8 + left.numel() * 8 + (k + m) * column_tile * 8
        )
    else:
        out.zero_()
        for start in range(0, k, k_tile):
            out.addmm_(
                a[:, start : start + k_tile].double(),
                b[start : start + k_tile].double(),
            )
        scheme = "FP64 sum of2048-wide K products, independent of GPU FP32 reduction"
        owned_double_tensor_bound = out.numel() * 8 + (m + n) * k_tile * 8
    return out, dict(
        scheme=scheme,
        seconds=time.monotonic() - started,
        owned_double_tensor_bound_bytes=owned_double_tensor_bound,
        reference_bytes=out.numel() * 8,
    )


def check(actual, expected):
    # Download BF16 only. Never allocate another full-vocabulary FP64 matrix.
    actual = actual.cpu()
    error2 = norm2 = maximum = 0.0
    for row in range(0, actual.shape[0], 16):
        for col in range(0, actual.shape[1], 4096):
            value = actual[row : row + 16, col : col + 4096].double()
            target = expected[row : row + 16, col : col + 4096]
            torch.testing.assert_close(value, target, rtol=0.015, atol=0.05)
            delta = value - target
            error2 += float(delta.square().sum())
            norm2 += float(target.square().sum())
            maximum = max(maximum, float(delta.abs().max()))
    relative = (error2 / max(norm2, 1e-24)) ** 0.5
    assert relative < 0.004, relative
    return dict(relative_l2=relative, max_abs=maximum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-label", default="training-n38-r2-layout")
    parser.add_argument("--role", choices=("forward", "input-backward"), required=True)
    args = parser.parse_args()
    assert Path(args.audit_label).name == args.audit_label
    torch.set_num_threads(4)
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1013"
    if args.role == "forward":
        left = torch.arange(27).reshape(3, 9).to(torch.bfloat16) / 32
        right = torch.arange(99).reshape(9, 11).to(torch.bfloat16) / 64
        independent = left.double() @ right.double()
        for role in ("forward", "input-backward"):
            tiled, _ = reference(left, right, role, column_tile=4, k_tile=4)
            torch.testing.assert_close(tiled, independent, atol=1e-12, rtol=1e-12)
        check(independent.to(torch.bfloat16), independent)
        for corrupt, expected in (
            (torch.full((3, 11), float("nan")), independent),
            (torch.full((3, 11), 0.001), torch.zeros((3, 11), dtype=torch.float64)),
        ):
            try:
                check(corrupt, expected)
            except AssertionError:
                pass
            else:
                raise AssertionError(
                    "Reference checker accepted nonfinite/corrupt output"
                )
    preset = Path("/workspace/rocblas-gemm-overrides.csv")
    assert (
        os.environ.get("ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH") == str(preset)
        and "LD_PRELOAD" not in os.environ
    )
    audit = args.output.parent.parent / args.audit_label
    rows = [
        r
        for r in json.loads((audit / "result.json").read_text())["layouts"]
        if r["role"] == args.role
    ]
    assert len(rows) == 1, rows
    row = rows[0]
    report = dict(
        status="running",
        checks=[],
        role=args.role,
        layout=row,
        results=[],
        gemm_preset_sha256=hashlib.sha256(preset.read_bytes()).hexdigest(),
        candidate_limit=32,
    )
    native = None
    try:
        fixture = audit / row["fixture"]
        with fixture.open("rb") as stream:
            assert (
                hashlib.file_digest(stream, "sha256").hexdigest()
                == row["fixture_sha256"]
            )
        cpu = torch.load(fixture, map_location="cpu", weights_only=True)
        a, b = cpu["a"].cuda(), cpu["b"].cuda()
        assert (
            list(a.stride()) == row["a_stride"] and list(b.stride()) == row["b_stride"]
        )
        assert row["a_offset"] == row["b_offset"] == 0
        assert (a.shape[0], a.shape[1], b.shape[1]) == (
            (1024, 2048, 248320) if args.role == "forward" else (1024, 248320, 2048)
        )
        expected, report["reference"] = reference(cpu["a"], cpu["b"], args.role)
        check(a @ b, expected)
        native = Gemm(a, b)
        report["rocblas_layout"] = native.layout
        available = native.solutions()
        report["available_solutions"] = available
        for index in [0] + available[:32]:
            native.call(index)
            numerical = check(native.out, expected)
            native.guards()
            timing = measure(lambda: native.call(index))
            report["results"].append(dict(index=index, **numerical, **timing))
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        del expected
        gc.collect()
        # Reuse the prior reference contract; bounded timing only after correctness.
        best = min(
            (r for r in report["results"] if r["index"] != 0), key=lambda r: r["gpu_ms"]
        )
        report["selected_candidate"] = best["index"]
        until = time.monotonic() + 2
        while time.monotonic() < until:
            native.call(0)
            torch.cuda.synchronize()
        samples = {"default": [], "candidate": []}
        for round_id in range(6):
            order = (
                ("default", "candidate")
                if round_id % 2 == 0
                else ("candidate", "default")
            )
            for name in order:
                index = 0 if name == "default" else best["index"]
                samples[name].append(measure(lambda: native.call(index), 3))
        native.guards()
        assert torch.equal(a.cpu(), cpu["a"]) and torch.equal(b.cpu(), cpu["b"])
        report["samples"] = samples
        report["medians"] = {
            name: statistics.median(s["gpu_ms"] for s in values)
            for name, values in samples.items()
        }
        report["paired_gpu_ratio"] = (
            report["medians"]["default"] / report["medians"]["candidate"]
        )
        report["max_process_rss_kib"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        report["checks"].append(
            "full-chunked-fp64-reference-output-guards-immutable-inputs-paired-timing"
        )
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        if native is not None:
            native.close()
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
