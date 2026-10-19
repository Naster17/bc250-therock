#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded native CK DL and retained GEMM CPU-reference checks."""
import argparse
import json
from pathlib import Path
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempt", default="r1")
    parser.add_argument("--legacy", action="store_true")
    args = parser.parse_args()
    ws = Path("/home/nik/bc250work/rocm-10.2-gfx1013")
    report = dict(status="running", checks=[])
    try:
        if args.legacy:
            for precision in ("fp32", "fp16", "int8"):
                binary = ws / "build/bc250-validation/ck" / ("gemm_dl_" + precision)
                command = [
                    str(binary),
                    "1",
                    "1",
                    "0",
                    "256",
                    "128",
                    "64",
                    "-1",
                    "-1",
                    "-1",
                ]
                result = subprocess.run(
                    command, check=True, capture_output=True, text=True, timeout=60
                )
                (args.output.parent / (precision + ".log")).write_text(
                    result.stdout + result.stderr
                )
                report["checks"].append(
                    dict(name=precision, status="passed", command=command)
                )
        else:
            binary = ws / ("build/bc250-coverage/ck-dl-n218-" + args.attempt) / "ck_dl"
            with (args.output.parent / "native-stderr.log").open("w") as error:
                result = subprocess.run(
                    [str(binary)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=error,
                    text=True,
                    timeout=120,
                )
            native = json.loads(result.stdout)
            assert native["status"] == "passed" and native["architecture"] == "gfx1013"
            assert (
                native["cases"] == 136 and native["streams"] == 2 and native["guards"]
            )
            assert len(native["checks"]) == 136
            (args.output.parent / "native-result.json").write_text(
                json.dumps(native, indent=2) + "\n"
            )
            report["checks"] = native["checks"]
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
