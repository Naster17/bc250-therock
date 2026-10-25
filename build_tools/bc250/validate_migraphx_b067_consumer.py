#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the relocated stable C API consumer and installed CLI without build trees."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--consumer", type=Path, required=True)
    args = parser.parse_args()
    sdk = Path(os.environ["BC250_MIGRAPHX_SDK"])
    assert not Path("/workspace/task/build-r1").exists()
    assert not Path(
        "/home/nik/bc250work/rocm-10.2-gfx1013/tmp/migraphx-b064/20261025-storage/mount/b066/build-r1"
    ).exists()
    models = sdk / "share/bc250-sdk/fixtures"
    report = dict(status="running", checks=[], sdk=str(sdk))
    try:
        process = subprocess.run(
            [str(args.consumer), str(models)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        (args.output.parent / "consumer-stdout.log").write_text(process.stdout)
        (args.output.parent / "consumer-stderr.log").write_text(process.stderr)
        assert process.returncode == 0, process.stderr
        result = json.loads(process.stdout.splitlines()[-1])
        assert result["status"] == "passed" and result["checks"] == 4
        report["checks"] = [f"public-c-api-reference-{i}" for i in range(4)]
        for name in ("pointwise", "constant"):
            result = subprocess.run(
                [
                    str(sdk / "bin/migraphx-driver"),
                    "read",
                    str(models / name / "model.onnx"),
                    "--onnx",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            (args.output.parent / (name + "-cli.log")).write_text(
                result.stdout + result.stderr
            )
            assert result.returncode == 0 and "@param" in result.stdout, result.stderr
            report["checks"].append(name + "-installed-cli-onnx-read")
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
