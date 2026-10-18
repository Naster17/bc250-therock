#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded FFT layouts, independent inverse inputs, gradients and plan reuse."""

import argparse
import itertools
import json
import math
from pathlib import Path
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--group", choices=("complex", "real", "contracts"), required=True
    )
    args = parser.parse_args()
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(10132140)
    arch = torch.cuda.get_device_properties(0).gcnArchName
    assert arch.split(":")[0] == "gfx1013", arch
    report = dict(status="running", arch=arch, torch=torch.__version__, checks=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def check(name, execute):
        start = time.monotonic()
        try:
            details = execute()
            torch.cuda.synchronize()
            row = dict(name=name, status="passed", details=details)
        except Exception as error:
            row = dict(name=name, status="failed", error=repr(error))
        row["seconds"] = time.monotonic() - start
        report["checks"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(row, flush=True)

    # Each layout is created independently on CPU/GPU, retaining actual device
    # strides and the leaf's unused elements in the backward comparison.
    def layout(shape, dtype, kind, device):
        rawshape = list(shape)
        if kind == "step":
            rawshape[-1] = shape[-1] * 2 + 3
        elif kind == "transpose":
            rawshape[-1], rawshape[-2] = rawshape[-2], rawshape[-1]
        elif kind == "expand":
            rawshape[0] = 1
        raw = torch.randn(rawshape, dtype=dtype)
        reference_dtype = torch.complex128 if dtype.is_complex else torch.float64
        cpu = raw.to(reference_dtype).requires_grad_()
        gpu = raw.to(device).detach().requires_grad_()
        assert cpu.is_leaf and gpu.is_leaf

        def view(value):
            if kind == "step":
                return value[..., 1 : 1 + 2 * shape[-1] : 2]
            if kind == "transpose":
                return value.transpose(-1, -2)
            if kind == "expand":
                return value.expand(shape)
            if kind == "conjugate":
                return value.conj()
            return value

        return cpu, gpu, view(cpu), view(gpu)

    def fixture(shape, dims, sizes, kind, dtype, operation, norm):
        cpu_leaf, gpu_leaf, cpu, gpu = layout(shape, dtype, kind, "cuda")
        original = gpu_leaf.detach().clone()
        op = getattr(torch.fft, operation)
        kw = dict(s=sizes, dim=dims, norm=norm)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            actual = op(gpu, **kw)
            expected = op(cpu, **kw)
            # Independent real/imaginary weights expose phase/sign errors that
            # a roundtrip or unweighted spectral energy alone could conceal.
            weight = torch.randn(expected.shape, dtype=expected.dtype)

            def loss(value, weights):
                linear = (value * weights.conj()).real.mean()
                return linear + 0.03 * value.abs().square().mean()

            loss(actual, weight.to(actual.dtype).cuda()).backward()
            loss(expected, weight).backward()
            # out= is exercised separately from autograd, into a sentinel view.
            guardshape = (*actual.shape[:-1], actual.shape[-1] * 2 + 3)
            guard = torch.full(guardshape, 123.25, dtype=actual.dtype, device="cuda")
            out = guard[..., 1 : 1 + actual.shape[-1] * 2 : 2]
            pointer = out.data_ptr()
            op(gpu.detach(), out=out, **kw)
            assert out.data_ptr() == pointer
        stream.synchronize()
        tolerance = 5e-4 if dtype in (torch.float32, torch.complex64) else 1e-10
        errors = {}
        for name, value, reference in (
            ("output", actual, expected),
            ("gradient", gpu_leaf.grad, cpu_leaf.grad),
            ("strided_out", out, expected),
        ):
            value = value.detach().cpu().to(reference.dtype)
            torch.testing.assert_close(value, reference, rtol=tolerance, atol=tolerance)
            errors[name] = (value - reference.detach()).abs().max().item()
        mask = torch.ones(guardshape, dtype=torch.bool)
        mask[..., 1 : 1 + actual.shape[-1] * 2 : 2] = False
        assert torch.all(guard.cpu()[mask] == 123.25)
        torch.testing.assert_close(gpu_leaf.detach(), original, rtol=0, atol=0)
        return dict(
            shape=shape,
            dims=dims,
            sizes=sizes,
            stride=gpu.stride(),
            storage_offset=gpu.storage_offset(),
            max_abs_errors=errors,
            sentinel_unchanged=True,
            input_unchanged=True,
            private_stream=True,
        )

    if args.group == "complex":
        specs = [
            ((3, 17), (-1,), None, "step"),
            ((2, 7, 9), (-2, -1), None, "transpose"),
            ((2, 7, 3, 5), (1, 3), None, "step"),
            ((2, 3, 5, 7), (0, 2, 3), None, "transpose"),
            ((2, 3, 5, 7, 3), (1, 2, 3, 4), None, "step"),
            ((3, 32), (-1,), None, "expand"),
            ((3, 19), (-1,), (17,), "conjugate"),
            ((3, 13), (-1,), (17,), "step"),
        ]
        for dtype, index, norm, op in itertools.product(
            (torch.complex64, torch.complex128),
            range(len(specs)),
            ("backward", "forward", "ortho"),
            ("fftn", "ifftn"),
        ):
            check(
                f"{op}_{dtype}_layout{index}_{norm}",
                lambda: fixture(*specs[index], dtype, op, norm),
            )
    elif args.group == "real":
        specs = [
            ((3, 17), (-1,), (17,), "step"),
            ((2, 45), (-1,), (45,), "expand"),
            ((2, 7, 13), (-2, -1), (7, 13), "transpose"),
            ((2, 7, 3, 5), (3, 1), (5, 7), "step"),
            ((2, 3, 5, 7, 3), (1, 2, 3, 4), (3, 5, 7, 3), "step"),
            ((3, 19), (-1,), (17,), "step"),
        ]
        for dtype, index, norm, inverse in itertools.product(
            (torch.float32, torch.float64),
            range(len(specs)),
            ("backward", "forward", "ortho"),
            (False, True),
        ):
            shape, dims, sizes, kind = specs[index]
            if inverse:
                # Random spectra independent of an RFFT: detects inverse/odd
                # size errors hidden by matching forward/inverse defects.
                shape = list(shape)
                for dim, size in zip(dims, sizes):
                    shape[dim] = size
                shape[dims[-1]] = sizes[-1] // 2 + 1
                dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
            op = "irfftn" if inverse else "rfftn"
            check(
                f"{op}_{dtype}_layout{index}_{norm}",
                lambda: fixture(shape, dims, sizes, kind, dtype, op, norm),
            )
    else:

        def independent_dft():
            n = 17
            x = torch.randn(2, n, dtype=torch.complex128)
            k = torch.arange(n, dtype=torch.float64)
            matrix = torch.exp((-2j * math.pi / n) * k[:, None] * k[None, :])
            expected = x @ matrix.T
            actual = torch.fft.fft(x.cuda()).cpu()
            torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
            return dict(max_abs_error=(actual - expected).abs().max().item())

        check("prime17_explicit_dft", independent_dft)

        def finite_difference():
            x = torch.randn(2, 17, dtype=torch.float64)
            weights = torch.randn(2, 9, dtype=torch.complex128)
            gpu = x.cuda().requires_grad_()

            def loss(value):
                return (
                    torch.fft.rfft(value, norm="ortho") * weights.to(value.device)
                ).real.sum()

            loss(gpu).backward()
            errors = []
            for row, col in ((0, 0), (0, 16), (1, 8)):
                plus, minus = x.clone(), x.clone()
                plus[row, col] += 1e-6
                minus[row, col] -= 1e-6
                numerical = (loss(plus) - loss(minus)) / 2e-6
                actual = gpu.grad[row, col].cpu()
                torch.testing.assert_close(actual, numerical, rtol=1e-7, atol=1e-8)
                errors.append(abs((actual - numerical).item()))
            return dict(coordinates=3, max_abs_error=max(errors))

        check("odd_rfft_finite_difference", finite_difference)

        def reuse_trace():
            cache = torch.backends.cuda.cufft_plan_cache
            capacity = cache.max_size
            # No clearing/resizing: persistent rocFFT AOT cache is separate.
            sizes = [cache.size]
            inputs = [
                torch.randn(3, 34, dtype=torch.complex64, device="cuda")[:, ::2]
                for _ in range(2)
            ]
            streams = [torch.cuda.Stream(), torch.cuda.Stream()]
            for stream in streams:
                stream.wait_stream(torch.cuda.current_stream())
            outputs = []
            for stream, value in zip(streams, inputs):
                with torch.cuda.stream(stream):
                    outputs.append(torch.fft.fft(value))
                stream.synchronize()
                sizes.append(cache.size)
            assert sizes[1] > sizes[0] and sizes[2] == sizes[1], sizes
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                for stream, value in zip(streams, inputs):
                    with torch.cuda.stream(stream):
                        value = value.detach().requires_grad_()
                        result = torch.fft.fft(value)
                        result.abs().square().mean().backward()
                    stream.synchronize()
            trace = args.output.parent / "fft-trace.json"
            prof.export_chrome_trace(str(trace))
            events = json.loads(trace.read_text())["traceEvents"]
            kernels = [e for e in events if e.get("cat") == "kernel"]
            fft_kernels = [
                e
                for e in kernels
                if any(s in e["name"].lower() for s in ("fft", "stockham", "bluestein"))
            ]
            assert fft_kernels, [e["name"] for e in kernels]
            for value, output in zip(inputs, outputs):
                torch.testing.assert_close(
                    output.cpu(), torch.fft.fft(value.cpu()), rtol=5e-4, atol=5e-4
                )
            assert cache.max_size == capacity
            return dict(
                cache_sizes=sizes,
                capacity=capacity,
                kernels=len(kernels),
                fft_kernel_names=sorted({e["name"] for e in fft_kernels}),
            )

        check("plan_reuse_two_private_streams_gpu_trace", reuse_trace)

    report["status"] = (
        "passed" if all(r["status"] == "passed" for r in report["checks"]) else "failed"
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    assert report["status"] == "passed"


if __name__ == "__main__":
    main()
