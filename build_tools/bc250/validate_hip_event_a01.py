#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Event-only reproducer: completed ordered events must yield stable timing."""

import argparse
import json
from pathlib import Path
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--expect-pass", action="store_true")
    args = ap.parse_args()
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1013"
    results = []
    x = torch.zeros(4096, device="cuda")
    for custom in [False, True]:
        stream = torch.cuda.Stream() if custom else torch.cuda.default_stream()
        with torch.cuda.stream(stream):
            for i in range(16):
                start, stop = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                start.record()
                for _ in range(4):
                    x.add_(1)
                stop.record()
                stop.synchronize()
                initial = start.elapsed_time(stop)
                # Keep the events alive while new work exercises signal recycling.
                for _ in range(8):
                    x.mul_(0.99)
                stream.synchronize()
                repeated = [start.elapsed_time(stop) for _ in range(4)]
                results.append(
                    dict(
                        custom=custom,
                        iteration=i,
                        initial_ms=initial,
                        repeated_ms=repeated,
                    )
                )
    bad = [
        r
        for r in results
        if r["initial_ms"] <= 0
        or min(r["repeated_ms"]) <= 0
        or any(abs(t - r["initial_ms"]) > 1e-5 for t in r["repeated_ms"])
    ]
    out = dict(samples=results, unstable_or_nonpositive=len(bad), tested=len(results))
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out), flush=True)
    if args.expect_pass:
        assert not bad


if __name__ == "__main__":
    main()
