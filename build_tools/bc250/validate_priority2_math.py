#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Bounded native gfx1013 FFT, solver and sparse numerical coverage through Torch."""

import argparse
import ctypes
import faulthandler
import json
from collections.abc import Callable
from pathlib import Path
import time


def main() -> None:
    faulthandler.enable()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--group",
        choices=("all", "blas", "fft", "solver", "sparse", "rng"),
        default="all",
    )
    args = parser.parse_args()
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(10132026)
    assert torch.cuda.is_available(), "GPU required"
    arch = torch.cuda.get_device_properties(0).gcnArchName
    assert arch.split(":")[0] == "gfx1013", arch
    torch.set_float32_matmul_precision("highest")
    started = time.monotonic()
    report = {
        "status": "running",
        "arch": arch,
        "torch": torch.__version__,
        "checks": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def check(name: str, execute: Callable, *, tolerance: float = 3e-4) -> None:
        start = time.monotonic()
        try:
            actual, expected = execute()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                actual.cpu().to(expected.dtype),
                expected,
                rtol=tolerance,
                atol=tolerance,
            )
            report["checks"].append(
                {"name": name, "status": "passed", "seconds": time.monotonic() - start}
            )
            print(f"PASS {name}", flush=True)
        except Exception as error:
            report["checks"].append(
                {"name": name, "status": "failed", "error": str(error)}
            )
            print(f"FAIL {name}: {error}", flush=True)
        save()

    if args.group in ("all", "blas"):
        for dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
        ):
            reference_dtype = torch.complex128 if dtype.is_complex else torch.float64
            tolerance = (
                0.05
                if dtype == torch.bfloat16
                else (
                    0.006
                    if dtype == torch.float16
                    else 4e-4 if dtype in (torch.float32, torch.complex64) else 1e-9
                )
            )
            left = torch.randn(3, 19, 11, dtype=dtype)
            right = torch.randn(3, 23, 11, dtype=dtype).transpose(-1, -2)
            gpu_left, gpu_right = left.cuda(), right.cuda()
            check(
                f"blas_bmm_strided_{dtype}",
                lambda: (
                    torch.bmm(gpu_left, gpu_right),
                    torch.bmm(left.to(reference_dtype), right.to(reference_dtype)),
                ),
                tolerance=tolerance,
            )
            bias = torch.randn(19, 23, dtype=dtype)
            check(
                f"blas_addmm_scaling_{dtype}",
                lambda: (
                    torch.addmm(
                        bias.cuda(), gpu_left[0], gpu_right[0], alpha=-0.7, beta=0.3
                    ),
                    torch.addmm(
                        bias.to(reference_dtype),
                        left[0].to(reference_dtype),
                        right[0].to(reference_dtype),
                        alpha=-0.7,
                        beta=0.3,
                    ),
                ),
                tolerance=tolerance,
            )
            check(
                f"blas_broadcast_matmul_{dtype}",
                lambda: (
                    gpu_left @ gpu_right[0],
                    left.to(reference_dtype) @ right[0].to(reference_dtype),
                ),
                tolerance=tolerance,
            )

    if args.group in ("all", "rng"):
        for distribution in ("uniform", "normal", "integer"):
            generator = torch.Generator(device="cuda").manual_seed(1013)

            def draw() -> torch.Tensor:
                if distribution == "normal":
                    return torch.randn(65536, device="cuda", generator=generator)
                if distribution == "integer":
                    return torch.randint(
                        0, 17, (65536,), device="cuda", generator=generator
                    )
                return torch.rand(65536, device="cuda", generator=generator)

            def replay() -> tuple[torch.Tensor, torch.Tensor]:
                state = generator.get_state()
                first = draw()
                generator.set_state(state)
                second = draw()
                return second, first.cpu()

            check(f"rng_{distribution}_state_replay", replay, tolerance=0)

            def statistics() -> tuple[torch.Tensor, torch.Tensor]:
                values = draw().double()
                if distribution == "normal":
                    observed = torch.stack((values.mean(), values.var(unbiased=False)))
                    expected = torch.tensor([0.0, 1.0], dtype=torch.float64)
                elif distribution == "integer":
                    assert values.min() >= 0 and values.max() < 17
                    observed = (
                        torch.bincount(values.long(), minlength=17).double()
                        / values.numel()
                    )
                    expected = torch.full((17,), 1 / 17, dtype=torch.float64)
                else:
                    assert values.min() >= 0 and values.max() < 1
                    observed = torch.stack((values.mean(), values.var(unbiased=False)))
                    expected = torch.tensor([0.5, 1 / 12], dtype=torch.float64)
                return observed, expected

            check(
                f"rng_{distribution}_distribution",
                statistics,
                tolerance=0.025 if distribution == "normal" else 0.004,
            )

    if args.group in ("all", "rng"):
        library = ctypes.CDLL("librocrand.so")
        library.rocrand_create_generator.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
        ]
        library.rocrand_destroy_generator.argtypes = [ctypes.c_void_p]
        library.rocrand_set_seed.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong]
        library.rocrand_generate_uniform.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        library.rocrand_generate_uniform_double.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        library.rocrand_generate_normal.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_float,
        ]

        def require(status: int) -> None:
            assert status == 0, f"rocRAND status {status}"

        for kind, name in (
            (401, "xorwow"),
            (402, "mrg32k3a"),
            (403, "mtgp32"),
            (404, "philox"),
            (405, "mrg31k3p"),
            (406, "lfsr113"),
        ):
            for dtype in (torch.float32, torch.float64):

                def uniform_replay() -> tuple[torch.Tensor, torch.Tensor]:
                    outputs = []
                    for _ in range(2):
                        generator = ctypes.c_void_p()
                        require(
                            library.rocrand_create_generator(
                                ctypes.byref(generator), kind
                            )
                        )
                        try:
                            require(library.rocrand_set_seed(generator, 1013))
                            output = torch.empty(65536, device="cuda", dtype=dtype)
                            generate = (
                                library.rocrand_generate_uniform_double
                                if dtype == torch.float64
                                else library.rocrand_generate_uniform
                            )
                            require(
                                generate(generator, output.data_ptr(), output.numel())
                            )
                            torch.cuda.synchronize()
                            assert output.min() > 0 and output.max() <= 1
                            assert abs(output.double().mean().item() - 0.5) < 0.01
                            assert (
                                abs(output.double().var(unbiased=False).item() - 1 / 12)
                                < 0.004
                            )
                            outputs.append(output.cpu())
                        finally:
                            require(library.rocrand_destroy_generator(generator))
                    return outputs[0], outputs[1]

                check(
                    f"rocrand_{name}_uniform_{dtype}_seed_replay",
                    uniform_replay,
                    tolerance=0,
                )

            def normal_statistics() -> tuple[torch.Tensor, torch.Tensor]:
                generator = ctypes.c_void_p()
                require(library.rocrand_create_generator(ctypes.byref(generator), kind))
                try:
                    require(library.rocrand_set_seed(generator, 1704))
                    output = torch.empty(65536, device="cuda", dtype=torch.float32)
                    require(
                        library.rocrand_generate_normal(
                            generator, output.data_ptr(), output.numel(), 0.0, 1.0
                        )
                    )
                    torch.cuda.synchronize()
                    values = output.double()
                    return torch.stack(
                        (values.mean(), values.var(unbiased=False))
                    ), torch.tensor([0.0, 1.0], dtype=torch.float64)
                finally:
                    require(library.rocrand_destroy_generator(generator))

            check(
                f"rocrand_{name}_normal_statistics", normal_statistics, tolerance=0.025
            )

    if args.group in ("all", "rng"):
        library.rocrand_set_quasi_random_generator_dimensions.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        for kind, name in (
            (501, "sobol32"),
            (502, "scrambled_sobol32"),
            (504, "sobol64"),
            (505, "scrambled_sobol64"),
        ):

            def quasi_replay() -> tuple[torch.Tensor, torch.Tensor]:
                outputs = []
                for _ in range(2):
                    generator = ctypes.c_void_p()
                    require(
                        library.rocrand_create_generator(ctypes.byref(generator), kind)
                    )
                    try:
                        require(
                            library.rocrand_set_quasi_random_generator_dimensions(
                                generator, 3
                            )
                        )
                        output = torch.empty(65538, device="cuda", dtype=torch.float64)
                        require(
                            library.rocrand_generate_uniform_double(
                                generator, output.data_ptr(), output.numel()
                            )
                        )
                        torch.cuda.synchronize()
                        assert output.min() >= 0 and output.max() <= 1
                        assert abs(output.mean().item() - 0.5) < 0.002
                        assert abs(output.var(unbiased=False).item() - 1 / 12) < 0.002
                        outputs.append(output.cpu())
                    finally:
                        require(library.rocrand_destroy_generator(generator))
                return outputs[0], outputs[1]

            check(f"rocrand_{name}_three_dimensions_replay", quasi_replay, tolerance=0)

    if args.group in ("all", "fft"):
        for dtype in (torch.float32, torch.float64):
            tolerance = 3e-4 if dtype == torch.float32 else 1e-9
            for shape in ((3, 17), (2, 45), (2, 7, 13)):
                # Slicing preserves genuinely strided GPU inputs, including odd lengths.
                source = torch.randn((*shape[:-1], shape[-1] * 2), dtype=dtype)
                cpu = source[..., ::2].double()
                gpu = source.cuda()[..., ::2]
                dimensions = (-1,) if len(shape) == 2 else (-2, -1)
                for norm in ("backward", "ortho", "forward"):
                    check(
                        f"rfftn_{dtype}_{shape}_{norm}",
                        lambda g=gpu, c=cpu, d=dimensions, n=norm: (
                            torch.fft.rfftn(g, dim=d, norm=n),
                            torch.fft.rfftn(c, dim=d, norm=n),
                        ),
                        tolerance=tolerance,
                    )
                check(
                    f"fft_roundtrip_{dtype}_{shape}",
                    lambda g=gpu, c=cpu, d=dimensions, s=shape: (
                        torch.fft.irfftn(
                            torch.fft.rfftn(g, dim=d), s=tuple(s[k] for k in d), dim=d
                        ),
                        c,
                    ),
                    tolerance=tolerance,
                )
            for length in (19, 32):
                cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
                data = torch.randn(2, length, dtype=cdtype)
                check(
                    f"complex_fft_{cdtype}_{length}",
                    lambda x=data: (
                        torch.fft.fft(x.cuda()),
                        torch.fft.fft(x.to(torch.complex128)),
                    ),
                    tolerance=tolerance,
                )

            def fft_gradient() -> tuple[torch.Tensor, torch.Tensor]:
                data = torch.randn(2, 31, dtype=dtype)
                cpu = data.double().clone().requires_grad_()
                gpu = data.cuda().requires_grad_()
                for value in (cpu, gpu):
                    torch.fft.rfft(value, norm="ortho").abs().square().sum().backward()
                return gpu.grad, cpu.grad

            check(f"rfft_backward_{dtype}", fft_gradient, tolerance=tolerance)

    if args.group in ("all", "solver"):
        for dtype in (torch.float32, torch.float64, torch.complex64, torch.complex128):
            reference_dtype = torch.complex128 if dtype.is_complex else torch.float64
            tolerance = 5e-4 if dtype in (torch.float32, torch.complex64) else 1e-9
            raw = torch.randn(2, 9, 9, dtype=dtype)
            matrix = raw @ raw.mH + 3 * torch.eye(9, dtype=dtype)
            # Noncontiguous transposed matrices and batched multiple RHS.
            matrix = matrix.transpose(-1, -2)
            rhs = torch.randn(2, 9, 3, dtype=dtype)
            cpu, right = matrix.to(reference_dtype), rhs.to(reference_dtype)
            gpu, gright = matrix.cuda(), rhs.cuda()
            check(
                f"solve_{dtype}",
                lambda: (
                    torch.linalg.solve(gpu, gright),
                    torch.linalg.solve(cpu, right),
                ),
                tolerance=tolerance,
            )
            check(
                f"cholesky_{dtype}",
                lambda: (torch.linalg.cholesky(gpu), torch.linalg.cholesky(cpu)),
                tolerance=tolerance,
            )
            check(
                f"eigvalsh_{dtype}",
                lambda: (torch.linalg.eigvalsh(gpu), torch.linalg.eigvalsh(cpu)),
                tolerance=tolerance,
            )
            check(
                f"svdvals_{dtype}",
                lambda: (torch.linalg.svdvals(gpu), torch.linalg.svdvals(cpu)),
                tolerance=tolerance,
            )
            check(
                f"inverse_residual_{dtype}",
                lambda: (
                    gpu @ torch.linalg.inv(gpu),
                    torch.eye(9, dtype=reference_dtype).expand(2, 9, 9),
                ),
                tolerance=tolerance,
            )
            rectangular = torch.randn(2, 13, 7, dtype=dtype)

            def qr_residual() -> tuple[torch.Tensor, torch.Tensor]:
                q, r = torch.linalg.qr(rectangular.cuda())
                # Reconstruction and orthogonality avoid arbitrary QR phase/sign choices.
                reconstructed = (q @ r).flatten()
                orthogonal = (q.mH @ q).flatten()
                expected = torch.cat(
                    (
                        rectangular.to(reference_dtype).flatten(),
                        torch.eye(7, dtype=reference_dtype).expand(2, 7, 7).flatten(),
                    )
                )
                return torch.cat((reconstructed, orthogonal)), expected

            check(f"qr_residual_{dtype}", qr_residual, tolerance=tolerance)

            def solve_gradient() -> tuple[torch.Tensor, torch.Tensor]:
                a, b = cpu.detach().requires_grad_(), right.detach().requires_grad_()
                ga, gb = gpu.detach().requires_grad_(), gright.detach().requires_grad_()
                for x, y in ((a, b), (ga, gb)):
                    torch.linalg.solve(x, y).abs().square().mean().backward()
                return torch.cat((ga.grad.flatten(), gb.grad.flatten())), torch.cat(
                    (a.grad.flatten(), b.grad.flatten())
                )

            check(f"solve_backward_{dtype}", solve_gradient, tolerance=tolerance)

        for dtype in (torch.float32, torch.float64):
            # A known-condition diagonal system provides an independent exact solution,
            # including scales separated by four orders of magnitude.
            diagonal = torch.logspace(0, 4, 11, dtype=dtype)
            matrix = torch.diag(diagonal)
            solution = torch.linspace(-0.5, 0.9, 22, dtype=dtype).reshape(11, 2)
            rhs = matrix @ solution
            check(
                f"solve_condition_1e4_{dtype}",
                lambda: (
                    torch.linalg.solve(matrix.cuda(), rhs.cuda()),
                    solution.double(),
                ),
                tolerance=5e-5 if dtype == torch.float32 else 1e-10,
            )

            def residual() -> tuple[torch.Tensor, torch.Tensor]:
                solved = torch.linalg.solve(matrix.cuda(), rhs.cuda())
                return (
                    matrix.cuda() @ solved - rhs.cuda()
                ) / rhs.norm(), torch.zeros_like(rhs).double()

            check(
                f"solve_condition_1e4_residual_{dtype}",
                residual,
                tolerance=1e-6 if dtype == torch.float32 else 1e-12,
            )

    if args.group in ("all", "sparse"):
        for dtype in (torch.float32, torch.float64):
            tolerance = 3e-4 if dtype == torch.float32 else 1e-9
            for shape in ((17, 13), (9, 21)):
                dense = torch.randn(shape, dtype=dtype)
                dense[torch.rand(shape) < 0.75] = 0
                dense[0] = 0  # Empty row exercises sparse row-pointer boundaries.
                right = torch.randn(shape[1], 10, dtype=dtype)
                for layout in (torch.sparse_coo, torch.sparse_csr):
                    sparse = dense.to_sparse(layout=layout)
                    gsparse = sparse.cuda()
                    # Transposed RHS gives a noncontiguous leading dimension.
                    padded = right.T.contiguous().cuda()
                    gright = padded.T
                    expected = dense.double() @ right.double()
                    check(
                        f"sparse_mm_{layout}_{dtype}_{shape}",
                        lambda: (torch.sparse.mm(gsparse, gright), expected),
                        tolerance=tolerance,
                    )
                    bias = torch.randn(shape[0], 10, dtype=dtype)
                    check(
                        f"sparse_addmm_{layout}_{dtype}_{shape}",
                        lambda: (
                            torch.sparse.addmm(
                                bias.cuda(), gsparse, gright, beta=0.3, alpha=-0.7
                            ),
                            0.3 * bias.double() - 0.7 * expected,
                        ),
                        tolerance=tolerance,
                    )

                    def dense_gradient() -> tuple[torch.Tensor, torch.Tensor]:
                        r = right.double().clone().requires_grad_()
                        g = gright.detach().requires_grad_()
                        (dense.double() @ r).square().sum().backward()
                        torch.sparse.mm(gsparse, g).square().sum().backward()
                        return g.grad, r.grad

                    check(
                        f"sparse_rhs_backward_{layout}_{dtype}_{shape}",
                        dense_gradient,
                        tolerance=tolerance,
                    )
        sparse_library = ctypes.CDLL("libhipsparse.so")
        pointer = ctypes.c_void_p
        integer = ctypes.c_int
        for function in (
            "hipsparseCreate",
            "hipsparseCreateMatDescr",
            "hipsparseCreateCsrsv2Info",
        ):
            getattr(sparse_library, function).argtypes = [ctypes.POINTER(pointer)]
        for function in (
            "hipsparseDestroy",
            "hipsparseDestroyMatDescr",
            "hipsparseDestroyCsrsv2Info",
        ):
            getattr(sparse_library, function).argtypes = [pointer]
        sparse_library.hipsparseSetMatFillMode.argtypes = [pointer, integer]
        sparse_library.hipsparseSetMatDiagType.argtypes = [pointer, integer]
        for prefix, dtype in (("S", torch.float32), ("D", torch.float64)):
            buffer_query = getattr(
                sparse_library, f"hipsparse{prefix}csrsv2_bufferSize"
            )
            analysis = getattr(sparse_library, f"hipsparse{prefix}csrsv2_analysis")
            solve = getattr(sparse_library, f"hipsparse{prefix}csrsv2_solve")
            common_types = [
                pointer,
                integer,
                integer,
                integer,
                pointer,
                pointer,
                pointer,
                pointer,
                pointer,
            ]
            buffer_query.argtypes = common_types + [ctypes.POINTER(integer)]
            analysis.argtypes = common_types + [integer, pointer]
            solve.argtypes = [
                pointer,
                integer,
                integer,
                integer,
                pointer,
                pointer,
                pointer,
                pointer,
                pointer,
                pointer,
                pointer,
                pointer,
                integer,
                pointer,
            ]
            for upper in (False, True):
                for unit in (False, True):
                    for transpose in (False, True):

                        def triangular() -> tuple[torch.Tensor, torch.Tensor]:
                            def ensure(status: int) -> None:
                                assert status == 0, f"hipSPARSE status {status}"

                            handle, descriptor, info = pointer(), pointer(), pointer()
                            ensure(sparse_library.hipsparseCreate(ctypes.byref(handle)))
                            ensure(
                                sparse_library.hipsparseCreateMatDescr(
                                    ctypes.byref(descriptor)
                                )
                            )
                            ensure(
                                sparse_library.hipsparseCreateCsrsv2Info(
                                    ctypes.byref(info)
                                )
                            )
                            try:
                                ensure(
                                    sparse_library.hipsparseSetMatFillMode(
                                        descriptor, int(upper)
                                    )
                                )
                                ensure(
                                    sparse_library.hipsparseSetMatDiagType(
                                        descriptor, int(unit)
                                    )
                                )
                                dense = torch.randn(17, 17, dtype=dtype) * 0.04
                                dense = (
                                    torch.triu(dense) if upper else torch.tril(dense)
                                )
                                dense.diagonal().fill_(1 if unit else 2)
                                csr = dense.to_sparse_csr()
                                values, rows, columns = (
                                    csr.values().cuda(),
                                    csr.crow_indices().int().cuda(),
                                    csr.col_indices().int().cuda(),
                                )
                                arguments = [
                                    handle,
                                    int(transpose),
                                    17,
                                    values.numel(),
                                    descriptor,
                                    values.data_ptr(),
                                    rows.data_ptr(),
                                    columns.data_ptr(),
                                    info,
                                ]
                                size = integer()
                                ensure(buffer_query(*arguments, ctypes.byref(size)))
                                workspace = torch.empty(
                                    max(size.value, 1), device="cuda", dtype=torch.uint8
                                )
                                ensure(analysis(*arguments, 1, workspace.data_ptr()))
                                expected = torch.linspace(-0.9, 0.7, 17, dtype=dtype)
                                matrix = dense.T if transpose else dense
                                rhs = (matrix @ expected).cuda()
                                output = torch.empty_like(rhs)
                                alpha = (
                                    ctypes.c_float(1)
                                    if dtype == torch.float32
                                    else ctypes.c_double(1)
                                )
                                ensure(
                                    solve(
                                        handle,
                                        int(transpose),
                                        17,
                                        values.numel(),
                                        ctypes.byref(alpha),
                                        descriptor,
                                        values.data_ptr(),
                                        rows.data_ptr(),
                                        columns.data_ptr(),
                                        info,
                                        rhs.data_ptr(),
                                        output.data_ptr(),
                                        1,
                                        workspace.data_ptr(),
                                    )
                                )
                                torch.cuda.synchronize()
                                # Check both the known solution and normalized residual.
                                residual = (matrix.cuda() @ output - rhs) / rhs.norm()
                                return torch.cat((output, residual)), torch.cat(
                                    (
                                        expected.double(),
                                        torch.zeros(17, dtype=torch.float64),
                                    )
                                )
                            finally:
                                ensure(sparse_library.hipsparseDestroyCsrsv2Info(info))
                                ensure(
                                    sparse_library.hipsparseDestroyMatDescr(descriptor)
                                )
                                ensure(sparse_library.hipsparseDestroy(handle))

                        check(
                            f"hipsparse_csrsv_{dtype}_upper{upper}_unit{unit}_transpose{transpose}",
                            triangular,
                            tolerance=3e-5 if dtype == torch.float32 else 1e-10,
                        )

        # Duplicate COO coordinates must contribute independently to their value gradients.
        indices = torch.tensor([[0, 0, 2, 4, 4], [1, 1, 3, 0, 6]])

        def duplicate_gradient() -> tuple[torch.Tensor, torch.Tensor]:
            values = torch.tensor(
                [0.2, -0.7, 1.2, 0.4, -0.3], dtype=torch.float64, requires_grad=True
            )
            gvalues = values.detach().float().cuda().requires_grad_()
            right = torch.randn(7, 3, dtype=torch.float32)
            cpu = torch.sparse_coo_tensor(indices, values, (5, 7))
            gpu = torch.sparse_coo_tensor(indices.cuda(), gvalues, (5, 7))
            torch.sparse.mm(cpu, right.double()).square().sum().backward()
            torch.sparse.mm(gpu, right.cuda()).square().sum().backward()
            return gvalues.grad, values.grad

        check("sparse_duplicate_coo_value_backward", duplicate_gradient, tolerance=5e-4)

    report["seconds"] = time.monotonic() - started
    report["passed"] = sum(item["status"] == "passed" for item in report["checks"])
    report["failed"] = len(report["checks"]) - report["passed"]
    report["status"] = "passed" if not report["failed"] else "failed"
    save()
    print(
        json.dumps(
            {key: report[key] for key in ("status", "passed", "failed", "seconds")}
        ),
        flush=True,
    )
    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
