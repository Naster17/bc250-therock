#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU-reference gates for a native gfx1013 PyTorch installation."""

import argparse
import copy
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="JSON result file")
    args = parser.parse_args()

    import torch
    import torch.nn.functional as functional
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one usable HIP device")
    properties = torch.cuda.get_device_properties(0)
    if properties.gcnArchName.split(":")[0] != "gfx1013":
        raise RuntimeError(f"Expected native gfx1013, got {properties.gcnArchName}")
    torch.set_num_threads(4)
    torch.manual_seed(1234)
    torch.set_float32_matmul_precision("highest")
    report = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "arch": properties.gcnArchName,
        "checks": [],
    }

    def passed(name: str) -> None:
        torch.cuda.synchronize()
        report["checks"].append(name)
        print(f"PASS {name}", flush=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    torch.cuda.nvtx.range_push("gfx1013_validation")
    torch.cuda.nvtx.mark("native_roctx_marker")
    torch.cuda.nvtx.range_pop()
    passed("roctx_markers")

    for dtype in (torch.float16, torch.bfloat16):
        for length in (1, 3, 17):
            indices = torch.arange(128) % length
            values = torch.full((128,), 0.5, dtype=dtype)
            for offset in (0, 1):
                cpu_storage = torch.full((length + 2,), 7.0, dtype=dtype)
                gpu_storage = cpu_storage.cuda()
                cpu_view = cpu_storage[offset : offset + length]
                gpu_view = gpu_storage[offset : offset + length]
                cpu_view.scatter_add_(0, indices, values)
                gpu_view.scatter_add_(0, indices.cuda(), values.cuda())
                torch.testing.assert_close(
                    gpu_storage.cpu(), cpu_storage, rtol=0, atol=0
                )
                passed(f"scatter_add_{dtype}_{length}_offset_{offset}")

                cpu_storage = torch.full((length * 3 + 2,), 7.0, dtype=dtype)
                gpu_storage = cpu_storage.cuda()
                cpu_view = cpu_storage[offset : offset + length * 3].view(length, 3)
                gpu_view = gpu_storage[offset : offset + length * 3].view(length, 3)
                updates = values[:, None].expand(-1, 3).contiguous()
                cpu_view.index_add_(0, indices, updates)
                gpu_view.index_add_(0, indices.cuda(), updates.cuda())
                torch.testing.assert_close(
                    gpu_storage.cpu(), cpu_storage, rtol=0, atol=0
                )
                passed(f"index_add_{dtype}_{length}_offset_{offset}")

            cpu_weight = torch.randn(length, 3, dtype=dtype, requires_grad=True)
            gpu_weight = cpu_weight.detach().cuda().requires_grad_()
            (functional.embedding(indices, cpu_weight).sum() * 0.5).backward()
            (functional.embedding(indices.cuda(), gpu_weight).sum() * 0.5).backward()
            torch.testing.assert_close(
                gpu_weight.grad.cpu(), cpu_weight.grad, rtol=0, atol=0
            )
            passed(f"embedding_gradient_{dtype}_{length}")

    # Compare against FP64 accumulation and cast only the final reference.
    # Low-precision tolerances allow a few output ULPs, including cancellation.
    precisions = (
        (torch.float32, 5e-4, 1e-4),
        (torch.float64, 1e-10, 1e-10),
        (torch.float16, 3e-3, 3e-3),
        (torch.bfloat16, 2e-2, 2e-2),
    )
    for dtype, relative, absolute in precisions:
        for shape in ((17, 31, 19), (64, 128, 96), (129, 65, 33)):
            m, k, n = shape
            a = torch.randn(m, k).to(dtype)
            b = torch.randn(n, k).to(dtype).transpose(0, 1)
            reference = (a.double() @ b.double()).to(dtype)
            result = (a.cuda() @ b.cuda()).cpu()
            torch.testing.assert_close(result, reference, rtol=relative, atol=absolute)
            passed(f"gemm_{dtype}_{m}_{k}_{n}")

    a, b = torch.randn(3, 17, 31), torch.randn(3, 31, 19)
    torch.testing.assert_close(
        torch.bmm(a.cuda(), b.cuda()).cpu(),
        torch.bmm(a.double(), b.double()).float(),
        rtol=5e-4,
        atol=1e-4,
    )
    passed("batched_gemm")

    for length in (17, 64, 1024):
        data = torch.randn(2, length)
        result = torch.fft.rfft(data.cuda()).cpu()
        reference = torch.fft.rfft(data.double()).to(result.dtype)
        torch.testing.assert_close(result, reference, rtol=1e-4, atol=1e-4)
        roundtrip = torch.fft.irfft(result.cuda(), n=length).cpu()
        torch.testing.assert_close(roundtrip, data, rtol=1e-4, atol=1e-4)
        passed(f"fft_{length}")

    # Dense forward, input gradients, and weight gradients exercise MIOpen.
    for groups in (1, 2):
        x = torch.randn(2, 4, 13, 11, dtype=torch.float64, requires_grad=True)
        w = torch.randn(8, 4 // groups, 3, 3, dtype=torch.float64, requires_grad=True)
        reference = functional.conv2d(x, w, padding=1, groups=groups)
        reference.square().mean().backward()
        gpu_x = x.detach().float().cuda().requires_grad_()
        gpu_w = w.detach().float().cuda().requires_grad_()
        result = functional.conv2d(gpu_x, gpu_w, padding=1, groups=groups)
        result.square().mean().backward()
        torch.testing.assert_close(
            result.cpu(), reference.float(), rtol=1e-3, atol=1e-4
        )
        torch.testing.assert_close(
            gpu_x.grad.cpu(), x.grad.float(), rtol=1e-3, atol=1e-4
        )
        torch.testing.assert_close(
            gpu_w.grad.cpu(), w.grad.float(), rtol=1e-3, atol=1e-4
        )
        passed(f"conv_forward_backward_groups_{groups}")

    q, k, v = (torch.randn(2, 4, 17, 32) for _ in range(3))
    reference = (
        torch.softmax(q.double() @ k.double().transpose(-2, -1) / 32**0.5, -1)
        @ v.double()
    )
    with sdpa_kernel(SDPBackend.MATH):
        result = functional.scaled_dot_product_attention(
            q.cuda(), k.cuda(), v.cuda()
        ).cpu()
    torch.testing.assert_close(result, reference.float(), rtol=5e-4, atol=1e-4)
    passed("sdpa_math")

    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64), torch.nn.ReLU(), torch.nn.Linear(64, 8)
    )
    gpu_model = copy.deepcopy(model).cuda()
    cpu_optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    gpu_optimizer = torch.optim.SGD(gpu_model.parameters(), lr=0.05)
    inputs, targets = torch.randn(64, 32), torch.randn(64, 8)
    gpu_inputs, gpu_targets = inputs.cuda(), targets.cuda()
    for step in range(30):
        cpu_optimizer.zero_grad()
        gpu_optimizer.zero_grad()
        cpu_loss = functional.mse_loss(model(inputs), targets)
        gpu_loss = functional.mse_loss(gpu_model(gpu_inputs), gpu_targets)
        torch.testing.assert_close(gpu_loss.cpu(), cpu_loss, rtol=5e-4, atol=1e-4)
        cpu_loss.backward()
        gpu_loss.backward()
        cpu_optimizer.step()
        gpu_optimizer.step()
    for cpu_parameter, gpu_parameter in zip(
        model.parameters(), gpu_model.parameters(), strict=True
    ):
        torch.testing.assert_close(
            gpu_parameter.cpu(), cpu_parameter, rtol=5e-4, atol=1e-4
        )
    passed("mlp_training_30_steps")
    print("TORCH_GFX1013_VALIDATION_PASS", flush=True)


if __name__ == "__main__":
    main()
