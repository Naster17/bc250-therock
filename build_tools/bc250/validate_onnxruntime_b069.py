#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the isolated ORT-style SDK boundary probe, not an installed ORT provider."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sdk = Path(os.environ["BC250_MIGRAPHX_SDK"])
    result = subprocess.run(
        [str(args.probe), str(sdk / "share/bc250-sdk/fixtures")],
        text=True,
        capture_output=True,
        timeout=400,
    )
    (args.output.parent / "probe-stdout.log").write_text(result.stdout)
    (args.output.parent / "probe-stderr.log").write_text(result.stderr)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.splitlines()[-1])
    assert report["status"] == "passed" and report["checks"] == 8
    assert report["native_architecture"] == "gfx1013:xnack-"
    report.update(
        checks=[
            f"{model}-{stream}-changed-input-{i}"
            for model in ("pointwise", "constant")
            for stream in ("default-stream", "private-stream")
            for i in (0, 1)
        ],
        ort_provider_built=False,
        sdk=str(sdk),
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
