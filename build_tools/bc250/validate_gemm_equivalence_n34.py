#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check output equivalence before considering another square forward candidate."""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch
from benchmark_gemm_n32 import Gemm, check_reference, measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    audit = args.output.parent.parent / "training-n34-r1-layout"
    row = next(
        r
        for r in json.loads((audit / "result.json").read_text())["layouts"]
        if r["b_stride"] == [1, 2048]
    )
    cpu = torch.load(audit / row["fixture"], weights_only=True, map_location="cpu")
    a, b = cpu["a"].cuda(), cpu["b"].cuda()
    expected = cpu["a"].double() @ cpu["b"].double()
    native = Gemm(a, b)
    report = dict(status="running", checks=[], results=[])
    try:
        original = native.call(0).clone()
        torch.testing.assert_close(a @ b, original, atol=0, rtol=0)
        for index in native.solutions():
            native.call(index)
            numeric = check_reference(native.out, expected)
            different = int((native.out != original).sum())
            relative = float(
                (native.out.float() - original.float()).norm() / original.float().norm()
            )
            report["results"].append(
                dict(
                    index=index,
                    different_elements=different,
                    relative_l2_to_default=relative,
                    cpu_reference=numeric,
                )
            )
        # Keep timing samples free of CPU reference/copy work after sustained warmup.
        exact = [r["index"] for r in report["results"] if not r["different_elements"]]
        timing = {}
        for index in exact:
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                for _ in range(10):
                    native.call(0)
                    native.call(index)
                torch.cuda.synchronize()
            samples = {"default": [], "candidate": []}
            for turn in range(6):
                order = (
                    ("default", "candidate")
                    if turn % 2 == 0
                    else ("candidate", "default")
                )
                for name in order:
                    solution = 0 if name == "default" else index
                    samples[name].append(measure(lambda: native.call(solution), 8))
            medians = {
                n: statistics.median(r["gpu_ms"] for r in rows)
                for n, rows in samples.items()
            }
            timing[str(index)] = dict(
                samples=samples,
                medians=medians,
                ratio=medians["default"] / medians["candidate"],
            )
        native.guards()
        assert torch.equal(a.cpu(), cpu["a"]) and torch.equal(b.cpu(), cpu["b"])
        report.update(
            status="passed",
            exact_indices=exact,
            exact_timings=timing,
            checks=["all-forward-solutions-cpu-reference-equivalence-guards"],
        )
    finally:
        native.close()
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
