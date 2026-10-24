#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Execute isolated pinned MIGraphX header probes; this is not a graph backend."""

import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent / "tmp/migraphx-b062/20261024-r2/tmp"
    report = dict(status="running", checks=[], variants={})
    try:
        for dpp in (1, 0):
            completed = subprocess.run(
                [str(root / ("probe-dpp" + str(dpp)))],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            result = json.loads(completed.stdout)
            assert result["status"] == "passed" and result["checks"] == 15
            assert (
                result["arch"].startswith("gfx1013:")
                and result["wavefront"] == 32
                and result["dpp"] == dpp
            )
            report["variants"][str(dpp)] = result
            # Each composite is checked inside the native executable, including
            # CPU-double oracles and all input/output padding guards.
            report["checks"] += [
                "dpp" + str(dpp) + "-" + name
                for name in [
                    "native-device",
                    "fp32-pointwise-stride-broadcast-guards",
                    "fp16-pointwise-stride-broadcast-guards",
                    *[
                        "reduce-cpu-double-guards-n" + str(n)
                        for n in (1, 17, 31, 32, 33, 63, 64, 127, 128, 129, 257, 513)
                    ],
                ]
            ]
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
