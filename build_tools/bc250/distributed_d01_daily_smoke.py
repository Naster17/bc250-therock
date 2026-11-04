#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Ordinary selected-release output/backward and consumer smoke, no user application."""
import argparse
import json
from pathlib import Path
import torch
import torch.distributed as dist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--consumers", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(1)
    assert (
        torch.cuda.get_device_properties(0).gcnArchName == "gfx1013:xnack-"
        and not dist.is_nccl_available()
    )
    cpu = [
        ((torch.arange(96).reshape(12, 8) % 19 - 9) / 16).double().requires_grad_(),
        ((torch.arange(32).reshape(8, 4) % 17 - 8) / 16).double().requires_grad_(),
    ]
    gpu = [x.detach().float().cuda().requires_grad_() for x in cpu]
    y = cpu[0] @ cpu[1]
    z = gpu[0] @ gpu[1]
    y.square().sum().backward()
    z.square().sum().backward()
    torch.cuda.synchronize()
    torch.testing.assert_close(z.cpu().double(), y, atol=0, rtol=0)
    for x, w in zip(cpu, gpu):
        torch.testing.assert_close(w.grad.cpu().double(), x.grad, atol=0, rtol=0)
    checks = [
        "native-daily-backend",
        "exact-forward",
        "exact-input-gradient",
        "exact-weight-gradient",
    ]
    consumers = []
    if a.consumers:
        import transformers, peft, accelerate

        consumers = [transformers.__version__, peft.__version__, accelerate.__version__]
        checks.append("frozen-consumer-imports")
    a.output.write_text(
        json.dumps(
            dict(
                status="passed",
                checks=checks,
                torch=torch.__version__,
                torch_path=torch.__file__,
                consumers=consumers,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
