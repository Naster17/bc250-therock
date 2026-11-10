#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Tiny CPU-first and native GPU output/backward checks, no user applications."""
import json
import torch

assert not torch.cuda.is_initialized()
torch.manual_seed(44)
x = torch.randn(17, 13, dtype=torch.float64, requires_grad=True)
w = torch.randn(13, 7, dtype=torch.float64, requires_grad=True)
reference = (x @ w).square().mean()
gx, gw = torch.autograd.grad(reference, (x, w))
assert not torch.cuda.is_initialized()
device_x = x.detach().float().cuda().requires_grad_()
device_w = w.detach().float().cuda().requires_grad_()
loss = (device_x @ device_w).square().mean()
loss.backward()
torch.testing.assert_close(loss.double().cpu(), reference, atol=1e-5, rtol=1e-5)
torch.testing.assert_close(device_x.grad.double().cpu(), gx, atol=1e-5, rtol=1e-5)
torch.testing.assert_close(device_w.grad.double().cpu(), gw, atol=1e-5, rtol=1e-5)
torch.cuda.synchronize()
print(
    json.dumps(
        dict(
            status="passed",
            checks=4,
            cpu_first_uninitialized=True,
            device=torch.cuda.get_device_name(),
            architecture=torch.cuda.get_device_properties(0).gcnArchName,
            torch=torch.__version__,
            loss=loss.item(),
            reference=reference.item(),
            allocated_bytes=torch.cuda.max_memory_allocated(),
        )
    ),
    flush=True,
)
