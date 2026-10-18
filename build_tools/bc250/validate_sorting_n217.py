#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run one bounded native primitive executable under the existing GPU guard."""

import argparse
import json
from pathlib import Path
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempt", default="r1")
    parser.add_argument(
        "--library", choices=("rocprim", "hipcub", "thrust", "legacy"), required=True
    )
    args = parser.parse_args()
    ws = Path("/home/nik/bc250work/rocm-10.2-gfx1013")
    directory = ws / "build/bc250-coverage"
    groups = (
        [
            (
                args.library,
                "sorting-n217-" + args.attempt,
                {"rocprim": 656, "hipcub": 656, "thrust": 648}[args.library],
            )
        ]
        if args.library != "legacy"
        else [
            (n, "priority2-primitives", c)
            for n, c in (("rocprim", 42), ("hipcub", 48), ("thrust", 48))
        ]
    )
    report = dict(status="running", checks=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for library, folder, count in groups:
            command = [str(directory / folder / library)]
            with (args.output.parent / (library + "-stderr.log")).open("w") as error:
                run = subprocess.run(
                    command,
                    check=True,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=error,
                    timeout=90,
                )
            result = json.loads(run.stdout)
            assert result["status"] == "passed" and result["architecture"] == "gfx1013"
            assert (
                result["cases"] == count
                and result["streams"] == 2
                and result["guards"] is True
            )
            (args.output.parent / (library + "-result.json")).write_text(
                json.dumps(result, indent=2) + "\n"
            )
            if args.library == "legacy":
                report["checks"].append(
                    dict(name=library, cases=count, status="passed")
                )
            else:
                assert len(result["checks"]) == count
                report["checks"].extend(result["checks"])
            print("PRIMITIVES_PASS", library, count, flush=True)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
