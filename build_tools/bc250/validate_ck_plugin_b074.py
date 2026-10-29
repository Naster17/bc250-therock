#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Paired forced public-Immediate solver timings, with CPU references and guards."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    ws = Path("/home/nik/bc250work/rocm-10.2-gfx1013")
    result = dict(status="running", checks=[], runs=[])
    try:
        for index in range(2):
            scratch = args.output.parent / ("fresh-" + str(index))
            scratch.mkdir()
            env = dict(
                os.environ,
                MIOPEN_LOG_LEVEL="3",
                MIOPEN_USER_DB_PATH=str(scratch / "db"),
                MIOPEN_CUSTOM_CACHE_DIR=str(scratch / "cache"),
            )
            cmd = [str(ws / "tmp/ck-plugin-b074/build-r1/probe-b074")]
            if index:
                cmd += ["--reverse"]
            proc = subprocess.run(
                cmd, env=env, capture_output=True, text=True, timeout=220
            )
            (scratch / "stdout.log").write_text(proc.stdout)
            (scratch / "stderr.log").write_text(proc.stderr)
            proc.check_returncode()
            run = json.loads(proc.stdout.strip().splitlines()[-1])
            assert run["status"] == "passed" and run["guards"] and len(run["rows"]) == 6
            assert run["dl_solver"] == 137 and run["direct_solver"] != 137
            for row in run["rows"]:
                assert len(row["windows"]) == 12
                row["median_stream_ms"] = [
                    statistics.median(
                        w["stream_interval_ms_per_call"]
                        for w in row["windows"]
                        if w["solver_index"] == j
                    )
                    for j in range(2)
                ]
                row["median_wall_ms"] = [
                    statistics.median(
                        w["wall_ms_per_call"]
                        for w in row["windows"]
                        if w["solver_index"] == j
                    )
                    for j in range(2)
                ]
                row["direct_over_dl_stream_ratio"] = (
                    row["median_stream_ms"][1] / row["median_stream_ms"][0]
                )
            result["runs"].append(run)
            result["checks"].append(
                "fresh-process-"
                + str(index)
                + "-forced-solvers-six-cpu-reference-paired-fixtures"
            )
        # Separate diagnostic execution: verbose logging must not contaminate timings.
        scratch = args.output.parent / "attribution"
        scratch.mkdir()
        env = dict(
            os.environ,
            MIOPEN_LOG_LEVEL="6",
            MIOPEN_USER_DB_PATH=str(scratch / "db"),
            MIOPEN_CUSTOM_CACHE_DIR=str(scratch / "cache"),
        )
        proc = subprocess.run(
            [str(ws / "tmp/ck-plugin-b074/build-r1/probe-b074")],
            env=env,
            capture_output=True,
            text=True,
            timeout=220,
        )
        (scratch / "stdout.log").write_text(proc.stdout)
        (scratch / "stderr.log").write_text(proc.stderr)
        proc.check_returncode()
        attributed = json.loads(proc.stdout.strip().splitlines()[-1])
        assert attributed["status"] == "passed" and len(attributed["rows"]) == 6
        for token in (
            "ConvHipImplicitGemmGroupFwdXdlops",
            "ConvHipDirectFwd",
            "kernel_name = DeviceGroupedConvFwdDl",
        ):
            assert token in proc.stderr, token
        result["attribution"] = attributed
        result["checks"].append(
            "separate-verbose-native-solver-and-dl-kernel-attribution"
        )
        import torch

        x = torch.arange(35, device="cuda", dtype=torch.float32).requires_grad_()
        x.square().sum().backward()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            x.grad.cpu(), 2 * torch.arange(35, dtype=torch.float32), rtol=0, atol=0
        )
        result["checks"].append("native-torch-backward-after-timing")
        result["status"] = "passed"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
