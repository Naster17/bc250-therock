#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Reference-check and time explicit private DL IDs against cached HIP-direct."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--build-attempt", choices=("r1", "r2", "r3"), required=True)
    args = p.parse_args()
    ws = Path("/home/nik/bc250work/rocm-10.2-gfx1013")
    plugin = ws / "tmp/ck-plugin-b073/build-r2/libMIOpenCKGroupedConv_gfx1013.so"
    result = dict(status="running", checks=[], runs=[])
    try:
        for index in range(3):
            diagnostic = index == 2
            scratch = args.output.parent / (
                "diagnostic" if diagnostic else "fresh-" + str(index)
            )
            scratch.mkdir()
            env = dict(
                os.environ,
                MIOPEN_LOG_LEVEL="6" if diagnostic else "3",
                MIOPEN_USER_DB_PATH=str(scratch / "db"),
                MIOPEN_CUSTOM_CACHE_DIR=str(scratch / "cache"),
            )
            cmd = [
                str(ws / f"tmp/ck-plugin-b075/build-{args.build_attempt}/probe-b075"),
                str(plugin),
            ]
            if index == 1:
                cmd += ["--reverse"]
            if diagnostic:
                cmd += ["--diagnostic"]
            proc = subprocess.run(
                cmd, env=env, capture_output=True, text=True, timeout=180
            )
            (scratch / "stdout.log").write_text(proc.stdout)
            (scratch / "stderr.log").write_text(proc.stderr)
            proc.check_returncode()
            r = json.loads(proc.stdout.strip().splitlines()[-1])
            assert (
                r["status"] == "passed"
                and r["guards"]
                and len(r["rows"]) == 40
                and len(r["kernel_ids"]) == 6
            )
            assert r["diagnostic"] == diagnostic
            assert r["checkpoints"] == (60 if diagnostic else 516)
            assert r["convolution_calls"] == (60 if diagnostic else 12972)
            for row in r["rows"]:
                nhwc = row["layout"] == "NHWC"
                assert len(row["windows"]) == (0 if diagnostic else 6 if nhwc else 12)
                assert (
                    row["workspace_bytes"] == 0 if nhwc else row["workspace_bytes"] > 0
                )
                if not diagnostic:
                    row["median_stream_ms"] = [
                        statistics.median(
                            w["stream_ms_per_call"]
                            for w in row["windows"]
                            if w["path"] == j
                        )
                        for j in range(1 if nhwc else 2)
                    ]
                    row["median_wall_ms"] = [
                        statistics.median(
                            w["wall_ms_per_call"]
                            for w in row["windows"]
                            if w["path"] == j
                        )
                        for j in range(1 if nhwc else 2)
                    ]
                    if not nhwc:
                        row["direct_over_dl_ratio"] = (
                            row["median_stream_ms"][1] / row["median_stream_ms"][0]
                        )
            if diagnostic:
                for kernel_id in r["kernel_ids"]:
                    assert "kernel_name = " + kernel_id in proc.stderr, kernel_id
                assert "kernel_name = MIOpenConvUni" in proc.stderr
                result["diagnostic"] = r
                result["checks"].append(
                    "six-explicit-IDs-and-HIP-direct-native-attribution"
                )
            else:
                result["runs"].append(r)
                result["checks"].append(
                    "opposite-order-"
                    + str(index)
                    + "-six-IDs-two-layouts-reference-paired-timing"
                )
        import torch

        x = torch.arange(35, device="cuda", dtype=torch.float32).requires_grad_()
        x.square().sum().backward()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            x.grad.cpu(), 2 * torch.arange(35, dtype=torch.float32), rtol=0, atol=0
        )
        result["checks"].append("native-torch-backward-after-private-invokers")
        result["status"] = "passed"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
