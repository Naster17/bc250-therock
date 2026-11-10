#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Predeclared small mixed first-backward/AdamW checks; no model or app inputs."""
import argparse
import faulthandler
import json
import math
from pathlib import Path
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.nn.functional as F

ATOL = 5e-5
RTOL = 5e-4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    faulthandler.enable()
    faulthandler.dump_traceback_later(45)
    assert not torch.cuda.is_initialized()
    generator = torch.Generator().manual_seed(args.seed)
    results = []

    def stage(name, event):
        print(json.dumps(dict(stage=name, event=event)), flush=True)

    def rand(*shape):
        # Both routes start from identical FP32-rounded values.
        return (torch.randn(*shape, generator=generator) * 0.1).double()

    def check(name, actual, expected):
        actual = actual.detach().cpu().to(expected.dtype)
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), name
        torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
        results.append(
            dict(
                name=name,
                elements=actual.numel(),
                max_abs=(actual - expected).abs().max().item(),
            )
        )

    def operation(name, inputs, cpu, gpu):
        stage(name, "begin")
        host = [v.clone().requires_grad_() for v in inputs]
        expected = cpu(*host)
        loss = expected.abs().square().mean()
        gradients = torch.autograd.grad(loss, host)
        device = [v.float().cuda().requires_grad_() for v in inputs]
        actual = gpu(*device)
        actual_loss = actual.abs().square().mean()
        actual_loss.backward()
        check(name + ".output", actual, expected)
        check(name + ".loss", actual_loss, loss.detach())
        for index, (tensor, gradient) in enumerate(zip(device, gradients)):
            check(f"{name}.gradient{index}", tensor.grad, gradient)
        torch.cuda.synchronize()
        stage(name, "passed")

    started = time.monotonic()
    operation(
        "matmul", [rand(32, 24), rand(24, 16)], lambda x, w: x @ w, lambda x, w: x @ w
    )
    operation(
        "conv",
        [rand(2, 2, 9, 11), rand(3, 2, 3, 3), rand(3)],
        lambda x, w, b: F.conv2d(x, w, b, padding=1),
        lambda x, w, b: F.conv2d(x, w, b, padding=1),
    )
    with sdpa_kernel(SDPBackend.MATH):
        operation(
            "sdpa",
            [rand(2, 2, 7, 8), rand(2, 2, 9, 8), rand(2, 2, 9, 6)],
            lambda q, k, v: ((q @ k.transpose(-2, -1)) / math.sqrt(8)).softmax(-1) @ v,
            lambda q, k, v: F.scaled_dot_product_attention(q, k, v),
        )
    operation(
        "fft", [rand(2, 33)], lambda x: torch.fft.rfft(x), lambda x: torch.fft.rfft(x)
    )
    stage("adamw", "begin")
    initial, x, target = rand(24, 16), rand(8, 24), rand(8, 16)
    host = initial.clone().requires_grad_()
    device = torch.nn.Parameter(initial.float().cuda())
    optimizer = torch.optim.AdamW(
        [device],
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        foreach=False,
    )
    m, v = torch.zeros_like(host), torch.zeros_like(host)
    for step in (1, 2):
        expected_loss = (x @ host - target).square().mean()
        (gradient,) = torch.autograd.grad(expected_loss, host)
        optimizer.zero_grad(set_to_none=True)
        actual_loss = (
            (x.float().cuda() @ device - target.float().cuda()).square().mean()
        )
        actual_loss.backward()
        check(f"adamw{step}.loss", actual_loss, expected_loss.detach())
        check(f"adamw{step}.gradient", device.grad, gradient)
        with torch.no_grad():
            m.mul_(0.9).add_(gradient, alpha=0.1)
            v.mul_(0.999).addcmul_(gradient, gradient, value=0.001)
            host.mul_(1 - 0.001 * 0.01)
            host.addcdiv_(
                m / (1 - 0.9**step), (v / (1 - 0.999**step)).sqrt() + 1e-8, value=-0.001
            )
        optimizer.step()
        state = optimizer.state[device]
        check(f"adamw{step}.parameter", device, host.detach())
        check(f"adamw{step}.first_moment", state["exp_avg"], m)
        check(f"adamw{step}.second_moment", state["exp_avg_sq"], v)
        assert state["step"].item() == step
    stage("adamw", "passed")
    torch.cuda.synchronize()
    record = dict(
        status="passed",
        seed=args.seed,
        checks=results,
        optimizer_steps=2,
        atol=ATOL,
        rtol=RTOL,
        architecture=torch.cuda.get_device_properties(0).gcnArchName,
        torch=torch.__version__,
        gpu_allocated_peak=torch.cuda.max_memory_allocated(),
        gpu_reserved_peak=torch.cuda.max_memory_reserved(),
        seconds=time.monotonic() - started,
    )
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    faulthandler.cancel_dump_traceback_later()
    print(json.dumps(dict(status="passed", checks=len(results))), flush=True)


if __name__ == "__main__":
    main()
