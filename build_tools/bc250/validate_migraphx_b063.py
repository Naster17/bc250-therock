#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Execute isolated pinned MIGraphX header probes; this is not a graph backend."""

import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent / "tmp/migraphx-b063/20261024-r4/tmp"
    scratch = root.parent.parent / "runtime-r2"
    assert scratch.is_dir() and os.access(scratch, os.W_OK)
    env = dict(os.environ, TMPDIR=str(scratch), TMP=str(scratch), TEMP=str(scratch))
    report = dict(status="running", checks=[], variants={}, scratch=str(scratch))
    try:
        for dpp in (1, 0):
            completed = subprocess.run(
                [str(root / ("probe-dpp" + str(dpp)))],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
            )
            lines = completed.stdout.splitlines()
            assert len(lines) == 2
            result = json.loads(lines[0])
            extra = json.loads(lines[1])
            assert extra["status"] == "passed"
            assert extra["additional_checks"] == (30 if dpp else 1)
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
            report["checks"] += (
                [
                    f"subwave-{w}-n{n}"
                    for w in (1, 2, 4, 8, 16, 32)
                    for n in (1, 17, 32, 33, 65)
                ]
                if dpp
                else ["upstream-no-dpp-block-regression"]
            )
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        for mode, label, count in (
            ("baseline", "migraphx-b062", 2),
            ("patched", "migraphx-b063", 8),
        ):
            source = Path(__file__).resolve().parent.parent / label / "source"
            run = subprocess.run(
                [str(root / "probe-jit"), str(source), mode],
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
            )
            (args.output.parent / ("hiprtc-" + mode + ".log")).write_text(run.stderr)
            assert run.returncode == 0, run.stderr
            result = json.loads(run.stdout)
            assert result["status"] == "passed" and result["checks"] == count
            report["checks"] += ["hiprtc-" + mode + "-" + str(i) for i in range(count)]
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
