#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bounded CPU-reference coverage of library-backed Torch operators on gfx1013."""

import argparse
import faulthandler
import json
from pathlib import Path
import time


def main() -> None:
    faulthandler.enable()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch.nn.functional as f

    torch.set_num_threads(4)
    torch.manual_seed(91013)
    assert torch.cuda.is_available()
    arch = torch.cuda.get_device_properties(0).gcnArchName
    assert arch.split(":")[0] == "gfx1013", arch
    torch.set_float32_matmul_precision("highest")
    report = {
        "arch": arch,
        "torch": torch.__version__,
        "checks": [],
        "status": "running",
    }
    start = time.monotonic()

    def check(name, operation, inputs, *, backward=False, tolerance=3e-4):
        cpu = [x.detach().clone().requires_grad_(backward) for x in inputs]
        gpu = [x.detach().cuda().requires_grad_(backward) for x in inputs]
        reference = operation(*cpu)
        result = operation(*gpu)
        torch.testing.assert_close(
            result.cpu(), reference, rtol=tolerance, atol=tolerance
        )
        if backward:
            # Nonuniform upstream gradients exercise actual indexing and backward kernels.
            grad = torch.linspace(-0.7, 0.9, reference.numel()).reshape(reference.shape)
            reference.backward(grad)
            result.backward(grad.cuda())
            for c, g in zip(cpu, gpu):
                assert c.grad is not None and g.grad is not None, name
                torch.testing.assert_close(
                    g.grad.cpu(), c.grad, rtol=tolerance, atol=tolerance
                )
        torch.cuda.synchronize()
        report["checks"].append(name)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"PASS {name}", flush=True)

    for dtype, tol in (
        (torch.float32, 4e-4),
        (torch.float64, 1e-10),
        (torch.float16, 5e-3),
        (torch.bfloat16, 4e-2),
    ):
        for m, k, n in ((1, 7, 3), (19, 23, 11)):
            # Device-side views preserve a nonzero offset and a noncontiguous leading stride.
            a = torch.randn(m + 2, k * 2 + 1).to(dtype)
            b = torch.randn(n + 1, k + 2).to(dtype)
            check(
                f"strided_gemm_{dtype}_{m}_{k}_{n}",
                lambda x, y: x[1 : 1 + m, 1 : 1 + 2 * k : 2] @ y[:n, 1 : 1 + k].T,
                [a, b],
                tolerance=tol,
            )
        x = torch.randn(3, 7, 19).to(dtype)
        check(
            f"reduce_transposed_{dtype}",
            lambda a: a.transpose(0, 2).sum(dim=1),
            [x],
            tolerance=tol,
        )
        check(
            f"softmax_offset_{dtype}",
            lambda a: f.softmax(a[:, :, 1:18:2], dim=-1),
            [x],
            tolerance=tol,
        )

    for rank in (1, 2, 3):
        spatial = {1: (19,), 2: (9, 11), 3: (5, 7, 9)}[rank]
        conv = getattr(f, f"conv{rank}d")
        for groups in (1, 2):
            x = torch.randn((2, 4, *spatial)) * 0.2
            w = torch.randn((6, 4 // groups, *((3,) * rank))) * 0.2
            check(
                f"conv{rank}d_stride_dilation_groups{groups}",
                lambda a, b, op=conv, g=groups: op(
                    a, b, stride=2, padding=2, dilation=2, groups=g
                ),
                [x, w],
                backward=True,
            )
    for layout in (torch.contiguous_format, torch.channels_last):
        x = torch.randn(2, 4, 9, 11).contiguous(memory_format=layout)
        w = torch.randn(6, 4, 3, 3).contiguous(memory_format=layout) * 0.1
        check(
            f"conv2d_layout_{layout}",
            lambda a, b: f.conv2d(a, b, padding=1),
            [x, w],
            backward=True,
        )
        check(
            f"batchnorm_training_{layout}",
            lambda a: f.batch_norm(a, None, None, training=True),
            [x],
            backward=True,
        )
        check(
            f"maxpool_{layout}", lambda a: f.max_pool2d(a, 3, 2, 1), [x], backward=True
        )
        check(
            f"avgpool_{layout}", lambda a: f.avg_pool2d(a, 3, 2, 1), [x], backward=True
        )
        check(
            f"bilinear_{layout}",
            lambda a: f.interpolate(
                a, size=(7, 13), mode="bilinear", align_corners=False
            ),
            [x],
            backward=True,
        )
    x = torch.randn(3, 5, 17)
    check(
        "layernorm_affine_backward",
        lambda a, w, b: f.layer_norm(a, (17,), w, b),
        [x, torch.randn(17), torch.randn(17)],
        backward=True,
    )
    check("groupnorm_backward", lambda a: f.group_norm(a, 1), [x], backward=True)
    check("gelu_backward", lambda a: f.gelu(a), [x], backward=True)
    for dtype, tol in (
        (torch.float32, 3e-4),
        (torch.float16, 5e-3),
        (torch.bfloat16, 4e-2),
    ):
        x = (
            torch.randn(2, 3, 9, 11)
            .to(dtype)
            .contiguous(memory_format=torch.channels_last)
        )
        for include in (True, False):
            check(
                f"avgpool_channels_last_padding_{dtype}_include{include}",
                lambda a, inc=include: f.avg_pool2d(
                    a, 3, 2, 1, ceil_mode=True, count_include_pad=inc
                ),
                [x],
                backward=True,
                tolerance=tol,
            )
    for length in (17, 63, 128):
        x = torch.randn(2, length)
        check(f"rfft_{length}", lambda a: torch.fft.rfft(a[:, 1:]), [x])
        check(
            f"irfft_roundtrip_{length}",
            lambda a, n=length: torch.fft.irfft(torch.fft.rfft(a), n=n),
            [x],
            backward=True,
        )
    for dtype, tol in ((torch.float32, 3e-4), (torch.float64, 1e-10)):
        a = torch.randn(13, 7, dtype=dtype)
        check(f"svd_values_{dtype}", torch.linalg.svdvals, [a], tolerance=tol)
        spd = a.T @ a + torch.eye(7, dtype=dtype) * 7
        check(f"eigenvalues_{dtype}", torch.linalg.eigvalsh, [spd], tolerance=tol)
        dense = torch.randn(11, 13, dtype=dtype)
        dense[dense.abs() < 1] = 0
        right = torch.randn(13, 5, dtype=dtype)
        for layout in (torch.sparse_csr, torch.sparse_coo):
            sparse = dense.to_sparse(layout=layout)
            actual = torch.sparse.mm(sparse.cuda(), right.cuda()).cpu()
            torch.testing.assert_close(actual, dense @ right, rtol=tol, atol=tol)
            report["checks"].append(f"sparse_{layout}_{dtype}")
    report.update(status="passed", seconds=time.monotonic() - start)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"OPERATOR_COVERAGE_PASS {len(report['checks'])} checks", flush=True)


if __name__ == "__main__":
    main()
