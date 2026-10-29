#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Execute the isolated installed-archive factory probes and native Torch coexistence."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--attempt", choices=("r1", "r2", "r3"), required=True)
    args = p.parse_args()
    ws = Path("/home/nik/bc250work/rocm-10.2-gfx1013")
    report = dict(status="running", checks=[], probes={})
    try:
        for mode in ("nodl", "dl"):
            proc = subprocess.run(
                [
                    str(
                        ws
                        / ("tmp/ck-plugin-b071/build-" + args.attempt)
                        / ("factory-" + mode)
                    )
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            (args.output.parent / (mode + "-stderr.log")).write_text(proc.stderr)
            (args.output.parent / (mode + "-stdout.log")).write_text(proc.stdout)
            result = json.loads(proc.stdout.strip().splitlines()[-1])
            assert result["status"] == "passed"
            if mode == "nodl":
                assert result["instances"] == result["executions"] == 0
            else:
                assert result["instances"] > 0 and result["executions"] >= 16
                assert result["fixtures"] == 16 and result["guards"]
                assert result["architecture"] == "gfx1013"
            report["probes"][mode] = result
            report["checks"].append(mode + "-factory-admission-and-reference")
        import torch

        assert torch.cuda.get_device_properties(0).gcnArchName == "gfx1013:xnack-"
        x = torch.arange(35, device="cuda", dtype=torch.float32).requires_grad_()
        x.square().sum().backward()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            x.grad.cpu(), 2 * torch.arange(35, dtype=torch.float32), rtol=0, atol=0
        )
        report["checks"].append("native-torch-backward-after-factory")
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
