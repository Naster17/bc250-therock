#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check FP16 raw factories and native first gradients without enabling a new plugin."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--build-attempt", choices=("r1", "r2", "r3", "r4", "r5"), required=True
    )
    args = p.parse_args()
    ws = Path("/home/nik/bc250work/rocm-10.2-gfx1013")
    r = dict(status="running", checks=[], probes={})
    try:
        for mode in ("nodl", "dl"):
            proc = subprocess.run(
                [
                    str(
                        ws
                        / f"tmp/ck-plugin-b079/build-{args.build_attempt}/factory-{mode}"
                    )
                ],
                capture_output=True,
                text=True,
                timeout=180,
            )
            (args.output.parent / (mode + "-stdout.log")).write_text(proc.stdout)
            (args.output.parent / (mode + "-stderr.log")).write_text(proc.stderr)
            proc.check_returncode()
            value = json.loads(proc.stdout.strip().splitlines()[-1])
            assert (
                value["status"] == "passed"
                and value["architecture"] == "gfx1013:xnack-"
            )
            if mode == "nodl":
                assert value["executions"] == value["fixtures"] == 0 and all(
                    x["instances"] == 0 for x in value["layouts"].values()
                )
            else:
                assert (
                    value["fixtures"] == 88
                    and value["executions"] > 88
                    and value["refusals"] > 0
                    and value["accumulator_executions"] > 0
                    and value["guards"]
                )
                assert (
                    value["unique_ids"] == 9
                    and len(value["kernel_counts"]) == 18
                    and all(n > 0 for n in value["kernel_counts"].values())
                )
                assert value["device_flags"] == dict(dot2=0, mac_f32=1, wave_size=32)
                assert value["accumulator_fixture"] == dict(
                    FP64_sum=256.25,
                    sequential_FP16_sum=256.0,
                    expected_FP16_output=256.25,
                )
            r["probes"][mode] = value
            r["checks"].append(mode + "-FP16-factory-reference-and-admission")
        import torch
        import torch.nn.functional as F

        torch.set_num_threads(1)
        torch.manual_seed(7901)
        assert torch.cuda.get_device_properties(0).gcnArchName == "gfx1013:xnack-"
        for groups in (1, 3):
            for f in (1, 3):
                values = [
                    (torch.randn(2, groups * 4, 5, 7) / 8).half(),
                    (torch.randn(groups * 4, 4, f, f) / 8).half(),
                    (torch.randn(groups * 4) / 8).half(),
                ]
                cpu = [x.double().requires_grad_() for x in values]
                gpu = [x.cuda().requires_grad_() for x in values]
                a = F.conv2d(*cpu, padding=f // 2, groups=groups)
                b = F.conv2d(*gpu, padding=f // 2, groups=groups)
                cot = (
                    torch.linspace(-0.5, 0.5, a.numel(), dtype=torch.float64)
                    .reshape_as(a)
                    .half()
                )
                (a * cot.double()).sum().backward()
                (b * cot.cuda()).sum().backward()
                torch.cuda.synchronize()
                torch.testing.assert_close(b.cpu().double(), a, atol=0.002, rtol=0.003)
                for ref, actual in zip(cpu, gpu):
                    torch.testing.assert_close(
                        actual.grad.cpu().double(), ref.grad, atol=0.003, rtol=0.006
                    )
                r["checks"].append(
                    f"existing-Torch-FP16-output-all-first-gradients-g{groups}-f{f}"
                )
        x = torch.arange(35, device="cuda", dtype=torch.float32).requires_grad_()
        x.square().sum().backward()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            x.grad.cpu(), 2 * torch.arange(35, dtype=torch.float32), rtol=0, atol=0
        )
        r["checks"].append("native-backward-after-FP16-factory")
        assert len(r["checks"]) == 7
        r["status"] = "passed"
    except BaseException as error:
        r.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(r, indent=2) + "\n")


if __name__ == "__main__":
    main()
