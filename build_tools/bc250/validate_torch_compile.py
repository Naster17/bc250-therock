# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validate native Triton dots and PyTorch Inductor forward/backward kernels."""

import argparse
import json
from pathlib import Path
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
