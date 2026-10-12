# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validate native Triton dots and PyTorch Inductor forward/backward kernels."""

import argparse
import json
import os
from pathlib import Path
import re
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triton-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Reuse pytest from the superbuild environment without changing Torch's
    # pinned dependencies or rebuilding its development container.
    sys.path.append("/opt/build-venv/lib/python3.12/site-packages")
    import pytest
    import torch
    import torch.nn.functional as functional
    import triton
    import triton.language as tl
    from torch._inductor import metrics

    assert torch.version.hip is not None
    assert triton.runtime.driver.active.get_current_target().arch == "gfx1013"
    torch.set_num_threads(2)
    torch.manual_seed(1704)
    report = {"torch": torch.__version__, "triton": triton.__version__, "checks": []}

    def passed(name: str) -> None:
        torch.cuda.synchronize()
        report["checks"].append(name)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"PASS {name}", flush=True)

    status = pytest.main(
        [
            "-s",
            "--tb=short",
            "--device=cuda",
            "-p",
            "no:cacheprovider",
            str(args.triton_source / "python/test/unit/language/test_core.py")
            + "::test_dot_gfx1013_scalar_fallback",
            str(args.triton_source / "python/test/unit/language/test_core.py")
            + "::test_bin_op[1-bfloat16-bfloat16-*]",
        ]
    )
    if status != 0:
        raise RuntimeError(
            f"Native Triton dot regression failed: pytest status {status}"
        )
    passed("triton_native_fp16_bf16_signed_int8_dots_and_bf16_multiply")

    @triton.jit
    def bf16_multiply(A, B, C, N: tl.constexpr):
        offsets = tl.program_id(0) * 128 + tl.arange(0, 128)
        a = tl.load(A + offsets, offsets < N, other=0)
        b = tl.load(B + offsets, offsets < N, other=0)
        tl.store(C + offsets, a * b, offsets < N)

    # BF16 products exactly halfway between representable results distinguish
    # round-to-nearest-even from truncation, including positive/negative ties.
    a = torch.tensor(
        [1.5, 1.5, 1.5, -1.5, 1.5, 0.0, -0.0, float("inf"), float("nan"), float("inf")],
        dtype=torch.bfloat16,
    ).repeat(13)
    b = torch.tensor(
        [
            1.0078125,
            1.0234375,
            1.0390625,
            1.0078125,
            -1.0078125,
            2.0,
            2.0,
            2.0,
            1.0,
            0.0,
        ],
        dtype=torch.bfloat16,
    ).repeat(13)
    reference = (a.double() * b.double()).bfloat16()
    output = torch.empty_like(a, device="cuda")
    compiled_kernel = bf16_multiply[(triton.cdiv(a.numel(), 128),)](
        a.cuda(), b.cuda(), output, a.numel(), num_warps=4
    )
    torch.testing.assert_close(output.cpu(), reference, rtol=0, atol=0, equal_nan=True)
    assert not re.search(r"\bv_(?:dot[24]|mfma|wmma)", compiled_kernel.asm["amdgcn"])
    passed("triton_bf16_multiply_exact_rtne_ties_and_nan")

    @triton.jit
    def masked_copy(X, Y, N: tl.constexpr):
        offsets = tl.program_id(0) * 128 + tl.arange(0, 128)
        values = tl.load(X + offsets, offsets < N, other=0)
        tl.store(Y + offsets, values * 2 + 7, offsets < N)

    buffer_operations = os.environ.get("AMDGCN_USE_BUFFER_OPS", "1") == "1"
    for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.int32):
        for length in (31, 129, 257):
            inputs = (torch.arange(length + 2) % 17 - 8).to(dtype)
            expected = torch.full_like(inputs, -17)
            expected[1:-1] = inputs[1:-1] * 2 + 7
            gpu_inputs = inputs.cuda()
            output = torch.full_like(gpu_inputs, -17)
            kernel = masked_copy[(triton.cdiv(length, 128),)](
                gpu_inputs[1:-1], output[1:-1], length, num_warps=4
            )
            torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
            assembly = kernel.asm["amdgcn"]
            if buffer_operations:
                assert re.search(r"\bbuffer_load_", assembly)
                assert re.search(r"\bbuffer_store_", assembly)
            else:
                assert not re.search(r"\bbuffer_(?:load|store)_", assembly)
            passed(f"triton_masked_memory_{dtype}_{length}_buffer_{buffer_operations}")

    def elementwise(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return functional.relu(x * y + 1)

    def normalize_and_project(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        normalized = functional.layer_norm(x, (37,))
        probabilities = torch.softmax(normalized, dim=-1)
        return (probabilities @ weight).square().mean()

    metrics.reset()
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        x, y = (torch.randn(17, 37).to(dtype) for _ in range(2))
        compiled = torch.compile(elementwise, fullgraph=True)
        result = compiled(x.cuda(), y.cuda()).cpu()
        tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-3
        torch.testing.assert_close(
            result, elementwise(x, y), rtol=tolerance, atol=tolerance
        )
        passed(f"inductor_elementwise_{dtype}")

    x = torch.randn(17, 37, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(37, 11, dtype=torch.float64, requires_grad=True)
    reference = normalize_and_project(x, weight)
    reference.backward()
    gpu_x = x.detach().float().cuda().requires_grad_()
    gpu_weight = weight.detach().float().cuda().requires_grad_()
    compiled = torch.compile(normalize_and_project, fullgraph=True)
    result = compiled(gpu_x, gpu_weight)
    result.backward()
    for actual, expected in (
        (result.cpu(), reference.float()),
        (gpu_x.grad.cpu(), x.grad.float()),
        (gpu_weight.grad.cpu(), weight.grad.float()),
    ):
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=1e-5)
    if metrics.generated_kernel_count == 0:
        raise RuntimeError("Inductor generated no kernels")
    report["inductor_generated_kernels"] = metrics.generated_kernel_count
    passed("inductor_layernorm_softmax_matmul_forward_backward")
    print("TORCH_COMPILE_GFX1013_VALIDATION_PASS", flush=True)


if __name__ == "__main__":
    main()
