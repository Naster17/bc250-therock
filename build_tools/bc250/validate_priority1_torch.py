#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bounded native gfx1013 eager/Inductor training and dynamic-shape coverage."""

import argparse
import ctypes
from collections.abc import Callable
import faulthandler
import json
from pathlib import Path
import time


def main() -> None:
    faulthandler.enable()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--group", choices=("all", "indexing", "training", "attention"), default="all"
    )
    parser.add_argument(
        "--with-hipblaslt",
        action="store_true",
        help="Also qualify explicit hipBLASLt selection; requires the optional scalar backend.",
    )
    parser.add_argument(
        "--hipblaslt-trace-library",
        type=Path,
        help="Test observer shared library, also loaded through LD_PRELOAD",
    )
    args = parser.parse_args()

    import torch
    import torch.nn.functional as f
    from torch._inductor import metrics
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch.set_num_threads(4)
    torch.manual_seed(10131704)
    assert torch.cuda.is_available(), "GPU required"
    arch = torch.cuda.get_device_properties(0).gcnArchName
    assert arch.split(":")[0] == "gfx1013", arch
    torch.set_float32_matmul_precision("highest")
    torch._dynamo.config.suppress_errors = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "arch": arch,
        "torch": torch.__version__,
        "checks": [],
    }
    start = time.monotonic()

    def save() -> None:
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def check(
        name: str,
        operation: Callable,
        batches: list[list[torch.Tensor]],
        *,
        reference: Callable | None = None,
        tolerance: float = 4e-4,
    ) -> None:
        # A recording backend delegates to real Inductor, not eager execution.
        graphs = []
        kernels_before = metrics.generated_kernel_count

        def backend(
            graph: torch.fx.GraphModule, inputs: list[torch.Tensor]
        ) -> Callable:
            graphs.append(str(graph.graph))
            return torch._inductor.compile(graph, inputs)

        compiled = torch.compile(
            operation, backend=backend, fullgraph=True, dynamic=True
        )
        for batch_index, inputs in enumerate(batches):
            # CPU double inputs begin with the exact quantized values sent to the GPU.
            cpu = [
                (
                    x.detach().double().requires_grad_(True)
                    if x.is_floating_point()
                    else x.detach().clone()
                )
                for x in inputs
            ]
            expected = (reference or operation)(*cpu)
            upstream = torch.linspace(
                -0.7, 0.9, expected.numel(), dtype=torch.float64
            ).reshape(expected.shape)
            expected.backward(upstream)
            for mode, execute in (("eager", operation), ("inductor", compiled)):
                gpu = [
                    x.detach().cuda().requires_grad_(x.is_floating_point())
                    for x in inputs
                ]
                actual = execute(*gpu)
                torch.testing.assert_close(
                    actual.cpu().double(),
                    expected.detach(),
                    rtol=tolerance,
                    atol=tolerance,
                )
                actual.backward(upstream.to(device="cuda", dtype=actual.dtype))
                for c, g in zip(cpu, gpu):
                    if c.requires_grad:
                        assert c.grad is not None and g.grad is not None, name
                        torch.testing.assert_close(
                            g.grad.cpu().double(),
                            c.grad,
                            rtol=tolerance,
                            atol=tolerance,
                        )
                torch.cuda.synchronize()
                label = f"{name}_shape{batch_index}_{mode}_forward_backward"
                report["checks"].append(label)
                save()
                print(f"PASS {label}", flush=True)
        assert graphs, f"{name}: compiler backend never called"
        generated = metrics.generated_kernel_count - kernels_before
        assert generated > 0, f"{name}: no Inductor kernels generated"
        report.setdefault("compilation", {})[name] = {
            "graphs": len(graphs),
            "generated_kernels": generated,
            "shapes": [[list(x.shape) for x in batch] for batch in batches],
            "fullgraph": True,
            "dynamic": True,
        }
        save()

    if args.group in ("all", "indexing"):

        def gather(a: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
            view = a[1:-1, 1::2]
            return torch.gather(view, 1, index).sin() + view[:, :1].square()

        batches = []
        for rows, width in ((3, 37), (5, 65)):
            indices = torch.arange(width + 4).repeat(rows, 1).remainder(width)
            batches.append([torch.randn(rows + 2, width * 2 + 1), indices])
        check("strided_duplicate_gather", gather, batches)

        def scatter(
            a: torch.Tensor, b: torch.Tensor, index: torch.Tensor
        ) -> torch.Tensor:
            return a.scatter_add(1, index, b).tanh()

        batches = []
        for width in (37, 65):
            idx = torch.arange(width * 2 + 3).repeat(3, 1).remainder(width)
            batches.append([torch.randn(3, width), torch.randn(idx.shape), idx])
        check("duplicate_scatter_add", scatter, batches)

        def index_add(
            a: torch.Tensor, b: torch.Tensor, index: torch.Tensor
        ) -> torch.Tensor:
            return a.index_add(0, index, b).square()

        batches = []
        for rows in (7, 11):
            idx = torch.arange(rows * 2 + 3).remainder(rows)
            batches.append([torch.randn(rows, 17), torch.randn(idx.numel(), 17), idx])
        check("duplicate_index_add", index_add, batches)

        def scatter_max(
            a: torch.Tensor, b: torch.Tensor, index: torch.Tensor
        ) -> torch.Tensor:
            return a.scatter_reduce(1, index, b, reduce="amax", include_self=False)

        batches = []
        for width in (17, 33):
            # Repeated equal maxima verify the backward gradient sharing rule.
            idx = torch.arange(width * 3).repeat(2, 1).remainder(width)
            batches.append([torch.randn(2, width), torch.ones(idx.shape), idx])
        check("scatter_amax_ties", scatter_max, batches)

        def embedding(weight: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
            return f.embedding(index, weight, padding_idx=0).cos()

        check(
            "embedding_duplicates_padding",
            embedding,
            [
                [torch.randn(23, 17), torch.arange(length).reshape(3, -1).remainder(23)]
                for length in (39, 75)
            ],
        )

    if args.group in ("all", "training"):

        def norm(
            a: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
        ) -> torch.Tensor:
            view = a[1:-1, 1::2]
            return f.layer_norm(view, (weight.numel(),), weight, bias).sin()

        for dtype, tol in (
            (torch.float32, 5e-4),
            (torch.float16, 6e-3),
            (torch.bfloat16, 5e-2),
        ):
            check(
                f"strided_layernorm_{dtype}",
                norm,
                [
                    [
                        torch.randn(rows + 2, width * 2 + 1).to(dtype),
                        torch.randn(width).to(dtype),
                        torch.randn(width).to(dtype),
                    ]
                    for rows, width in ((3, 37), (5, 65))
                ],
                tolerance=tol,
            )

        def mlp(a: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return f.layer_norm(f.gelu(f.linear(a, w, b)), (w.shape[0],)).square()

        check(
            "training_mlp",
            mlp,
            [
                [
                    torch.randn(rows, 17) * 0.2,
                    torch.randn(23, 17) * 0.2,
                    torch.randn(23) * 0.1,
                ]
                for rows in (3, 7)
            ],
            tolerance=6e-4,
        )

    if args.group in ("all", "attention"):
        for kind, dtype, tol in (
            ("causal", torch.float32, 5e-4),
            ("boolean_fully_masked", torch.float32, 5e-4),
            ("additive", torch.float16, 6e-3),
            ("gqa", torch.bfloat16, 5e-2),
        ):
            causal = kind == "causal"
            gqa = kind == "gqa"
            strided = kind == "boolean_fully_masked"

            def attention(q, k, v, mask, *, causal=causal, gqa=gqa, strided=strided):
                if strided:
                    q, k, v = q[..., 1::2], k[..., 1::2], v[..., 1::2]
                return f.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None if causal else mask,
                    is_causal=causal,
                    enable_gqa=gqa,
                    dropout_p=0.0,
                )

            def reference(q, k, v, mask, *, causal=causal, gqa=gqa, strided=strided):
                if strided:
                    q, k, v = q[..., 1::2], k[..., 1::2], v[..., 1::2]
                if gqa:
                    repeats = q.shape[1] // k.shape[1]
                    k = k.repeat_interleave(repeats, dim=1)
                    v = v.repeat_interleave(repeats, dim=1)
                scores = (q @ k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
                if causal:
                    allowed = torch.ones(
                        q.shape[-2], k.shape[-2], dtype=torch.bool
                    ).tril()
                    scores = scores.masked_fill(~allowed, -float("inf"))
                elif mask.dtype == torch.bool:
                    scores = scores.masked_fill(~mask, -float("inf"))
                else:
                    scores = scores + mask
                # Avoid undefined softmax derivatives for fully masked rows.
                fully_masked = torch.isneginf(scores).all(dim=-1, keepdim=True)
                safe = scores.masked_fill(fully_masked, 0)
                weights = safe.softmax(dim=-1).masked_fill(fully_masked, 0)
                return weights @ v

            batches = []
            for nq, nk in ((17, 23), (29, 37)):
                nh = 4 if gqa else 2
                q = torch.randn(1, nh, nq, 32 if strided else 16).to(dtype) * 0.2
                k = torch.randn(1, 2, nk, 32 if strided else 16).to(dtype) * 0.2
                v = torch.randn(1, 2, nk, 38 if strided else 19).to(dtype) * 0.2
                mask = torch.ones(nq, nk, dtype=torch.bool)
                if kind == "boolean_fully_masked":
                    mask[:, 1::3] = False
                    mask[3] = False
                elif kind in ("additive", "gqa"):
                    mask = torch.linspace(-0.3, 0.1, nq * nk).reshape(nq, nk).to(dtype)
                batches.append([q, k, v, mask])
            # Explicit math backend avoids selecting unqualified matrix-instruction kernels.
            with sdpa_kernel(SDPBackend.MATH):
                check(
                    f"attention_{kind}_{dtype}",
                    attention,
                    batches,
                    reference=reference,
                    tolerance=tol,
                )

    if args.with_hipblaslt:
        assert (
            args.hipblaslt_trace_library
        ), "Supply the test observer to verify actual Lt dispatch"
        trace = ctypes.CDLL(str(args.hipblaslt_trace_library.resolve()))
        counter = trace.bc250_hipblaslt_successful_calls
        counter.restype = ctypes.c_ulonglong
        report["hipblaslt_dispatch"] = {}
        library = torch.backends.cuda.preferred_blas_library
        previous = library()
        assert previous == torch._C._BlasBackend.Cublas, "Default must remain hipBLAS"
        try:
            library("hipblaslt")
            assert library() == torch._C._BlasBackend.Cublaslt, "Lt request fell back"
            for dtype, tol in (
                (torch.float32, 4e-4),
                (torch.float16, 6e-3),
                (torch.bfloat16, 5e-2),
            ):
                a = torch.randn(21, 49).to(dtype) * 0.2
                b = torch.randn(14, 27).to(dtype) * 0.2
                bias = torch.randn(13).to(dtype) * 0.1
                operations = (
                    (
                        "offset_strided_mm",
                        lambda x, y, z: x[1:20, 1:47:2] @ y[1:14, 2:25].T,
                    ),
                    (
                        "bias_addmm",
                        lambda x, y, z: torch.addmm(
                            z, x[1:20, 1:47:2], y[1:14, 2:25].T
                        ),
                    ),
                    (
                        "linear_bias",
                        lambda x, y, z: f.linear(x[1:20, 1:47:2], y[1:14, 2:25], z),
                    ),
                    (
                        "strided_bmm",
                        lambda x, y, z: torch.bmm(
                            x[1:19, 1:47:2].reshape(3, 6, 23),
                            y[1:13, 2:25].reshape(3, 4, 23).transpose(1, 2),
                        ),
                    ),
                )
                for label, operation in operations:
                    before = counter()
                    cpu = [x.double().requires_grad_(True) for x in (a, b, bias)]
                    gpu = [x.cuda().requires_grad_(True) for x in (a, b, bias)]
                    expected = operation(*cpu)
                    actual = operation(*gpu)
                    torch.testing.assert_close(
                        actual.cpu().double(), expected, rtol=tol, atol=tol
                    )
                    gradient = torch.linspace(
                        -0.7, 0.9, expected.numel(), dtype=torch.float64
                    ).reshape(expected.shape)
                    expected.backward(gradient)
                    actual.backward(gradient.to(device="cuda", dtype=dtype))
                    for c, g in zip(cpu, gpu):
                        if c.grad is None:
                            assert g.grad is None
                        else:
                            torch.testing.assert_close(
                                g.grad.cpu().double(), c.grad, rtol=tol, atol=tol
                            )
                    torch.cuda.synchronize()
                    assert library() == torch._C._BlasBackend.Cublaslt
                    name = f"hipblaslt_{label}_{dtype}_forward_backward"
                    calls = counter() - before
                    assert calls > 0, f"{name}: no successful hipBLASLt execution"
                    report["hipblaslt_dispatch"][name] = calls
                    report["checks"].append(name)
                    save()
                    print(f"PASS {name}", flush=True)
        finally:
            library(previous)
        assert library() == torch._C._BlasBackend.Cublas

    report.update(status="passed", seconds=time.monotonic() - start)
    save()
    print(f"PRIORITY1_TORCH_PASS {len(report['checks'])} checks", flush=True)


if __name__ == "__main__":
    main()
