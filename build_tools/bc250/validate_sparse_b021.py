#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded GPU CSR fallback: independent dense/analytic/FD first derivatives."""

import argparse
import itertools
import json
from pathlib import Path
from unittest.mock import patch

import torch
import bc250_sparse as sparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(1013021)
    assert Path(sparse.__file__).is_relative_to(
        "/opt/torch-venv/lib/python3.12/site-packages"
    )
    assert sparse.__version__ == "0.1.0"
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1013"
    report = dict(status="running", checks=[], results=[], module_file=sparse.__file__)

    def persist():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def passed(name, **details):
        report["checks"].append(name)
        report["results"].append(dict(name=name, **details))
        persist()

    def close(actual, expected, dtype):
        low = dtype in (torch.float32, torch.complex64)
        torch.testing.assert_close(
            actual.detach().cpu().to(expected.dtype),
            expected,
            rtol=3e-4 if low else 2e-10,
            atol=3e-6 if low else 2e-12,
        )
        assert actual.is_cuda

    def fixture(dtype, index, n, nrhs, layout, amp):
        rd = torch.complex128 if dtype.is_complex else torch.float64
        mask = torch.rand(n, n) < 0.19
        mask.fill_diagonal_(True)
        if n > 1:
            mask[0, -1] = True
        rows, cols = mask.nonzero(as_tuple=True)
        nnz = len(rows)
        values = torch.randn(nnz, dtype=dtype) * 0.1
        values[rows == cols] += 2
        if n > 1:
            values[(rows == 0) & (cols == n - 1)] = (
                0  # stored zero still has a derivative
            )
        crow = torch.cat(
            (
                torch.zeros(1, dtype=index),
                torch.bincount(rows, minlength=n).cumsum(0).to(index),
            )
        )
        col = cols.to(index)
        if layout == "step":
            vraw = torch.full((2 * nnz + 3,), 0.25, dtype=dtype)
            vraw[1 : 1 + 2 * nnz : 2] = values
            select = lambda t: t[1 : 1 + 2 * nnz : 2]
            braw = torch.randn(2 * n + 3, 2 * nrhs + 3, dtype=dtype)
            select_b = lambda t: t[1 : 1 + 2 * n : 2, 1 : 1 + 2 * nrhs : 2]
        else:
            vraw = values
            select = lambda t: t
            braw = torch.randn(n, nrhs, dtype=dtype)
            select_b = lambda t: t
        vg = vraw.cuda().detach().requires_grad_()
        bg = braw.cuda().detach().requires_grad_()
        vc = vraw.to(rd).requires_grad_()
        bc = braw.to(rd).requires_grad_()
        bview = lambda t: select_b(t).squeeze(1) if nrhs == 1 else select_b(t)
        cpu_a = torch.zeros((n, n), dtype=rd).index_put((rows, cols), select(vc))
        ref = torch.linalg.solve(cpu_a, bview(bc))
        weights = torch.randn(ref.shape, dtype=dtype).to(rd)
        loss = lambda x, w: (x * w.conj()).real.sum() + 0.03 * x.abs().square().sum()
        gc = torch.autograd.grad(loss(ref, weights), (vc, bc))
        before_v, before_b = vg.detach().clone(), bg.detach().clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=amp
        ):
            x = sparse.solve_csr(crow.cuda(), col.cuda(), select(vg), bview(bg))
            gg = torch.autograd.grad(loss(x, weights.to(dtype).cuda()), (vg, bg))
        torch.cuda.current_stream().wait_stream(stream)
        close(x, ref, dtype)
        for actual, expected in zip(gg, gc):
            close(actual, expected, dtype)
            assert torch.equal(
                actual.detach().cpu()[expected == 0],
                torch.zeros_like(actual.detach().cpu()[expected == 0]),
            )
        assert x.dtype == dtype and x.device == vg.device
        assert torch.equal(vg.detach(), before_v) and torch.equal(bg.detach(), before_b)
        dx = x.detach().cpu().to(rd)
        residual = float((cpu_a.detach() @ dx - bview(bc).detach()).abs().max())
        passed(
            f"matrix-{dtype}-{index}-{n}-{nrhs}-{layout}-amp{amp}",
            output_max_abs=float((dx - ref.detach()).abs().max()),
            residual_max_abs=residual,
        )
        return x, gg

    try:
        for config in itertools.product(
            (torch.float32, torch.float64, torch.complex64, torch.complex128),
            (torch.int32, torch.int64),
            (1, 7, 33),
            (1, 3),
            ("regular", "step"),
            (False, True),
        ):
            fixture(*config)
        for dtype, index in itertools.product(
            (torch.float32, torch.float64, torch.complex64, torch.complex128),
            (torch.int32, torch.int64),
        ):
            # Row-swapped invertible matrix forces LU pivoting; complex non-Hermitian.
            d = torch.tensor([[0, 2, 0], [3, 0, 1], [0, 1, 4]], dtype=dtype)
            if dtype.is_complex:
                d[0, 1] += 0.3j
            a0 = d.to_sparse_csr()
            v = a0.values().cuda().requires_grad_()
            b = torch.tensor(
                [[1, 2], [3, 4], [5, 6]], dtype=dtype, device="cuda", requires_grad=True
            )
            crow, col = (
                a0.crow_indices().to(index).cuda(),
                a0.col_indices().to(index).cuda(),
            )
            x = sparse.solve_csr(crow, col, v, b)
            w = torch.tensor(
                [[0.2, 0.5], [0.7, 0.3], [0.1, 0.4]], dtype=dtype, device="cuda"
            )
            gv, gb = torch.autograd.grad((x * w.conj()).real.sum(), (v, b))
            rd = torch.complex128 if dtype.is_complex else torch.float64
            da = d.to(rd)
            dr = b.detach().cpu().to(rd)
            xr = torch.linalg.solve(da, dr)
            adj = torch.linalg.solve(da.mH, w.cpu().to(rd))
            fullga = -(adj @ xr.mH)
            close(x, xr, dtype)
            close(gb, adj, dtype)
            close(gv, fullga[d != 0], dtype)
            # The CSR wrapper supports RHS gradients for all four dtypes.
            a = torch.sparse_csr_tensor(crow, col, v.detach(), size=(3, 3))
            x2 = sparse.spsolve(a, b)
            g2 = torch.autograd.grad((x2 * w.conj()).real.sum(), b)[0]
            close(x2, xr, dtype)
            close(g2, adj, dtype)
            if not dtype.is_complex:
                v2 = v.detach().requires_grad_()
                a2 = torch.sparse_csr_tensor(crow, col, v2, size=(3, 3))
                y = sparse.spsolve(a2, b)
                g = torch.autograd.grad((y * w.conj()).real.sum(), v2)[0]
                close(g, fullga[d != 0], dtype)
            passed(f"pivot-analytic-wrapper-{dtype}-{index}")
        # Diagonal systems admit independent analytic references at larger sizes
        # and the maximum RHS count, without running a second dense solver.
        sizes = [
            (dt, n, 1)
            for dt in (torch.float32, torch.float64, torch.complex64, torch.complex128)
            for n in (65, 257)
        ]
        sizes += [(torch.float32, 1024, 1)]
        sizes += [
            (dt, 3, 64)
            for dt in (torch.float32, torch.float64, torch.complex64, torch.complex128)
        ]
        for dt, n, k in sizes:
            crow = torch.arange(n + 1, device="cuda", dtype=torch.int64)
            col = torch.arange(n, device="cuda", dtype=torch.int64)
            val = (
                torch.linspace(2, 3, n, device="cuda").to(dt)
                + (0.2j if dt.is_complex else 0)
            ).requires_grad_()
            rhs = torch.ones(n, k, device="cuda", dtype=dt, requires_grad=True)
            x = sparse.solve_csr(crow, col, val, rhs)
            gv, gb = torch.autograd.grad(x.abs().square().sum(), (val, rhs))
            rd = torch.complex128 if dt.is_complex else torch.float64
            v = val.detach().cpu().to(rd)
            r = rhs.detach().cpu().to(rd)
            expected = r / v[:, None]
            close(x, expected, dt)
            adj = 2 * expected / v.conj()[:, None]
            close(gb, adj, dt)
            close(gv, -(adj * expected.conj()).sum(dim=1), dt)
            passed(f"diagonal-analytic-{dt}-{n}-{k}")
        # Inclusive budget boundary on a 2x2 system.
        a = torch.eye(2, device="cuda").to_sparse_csr()
        b = torch.ones(2, device="cuda")
        budget = 4 * (8 * 4 + 6 * 2)
        x = sparse.spsolve(a, b, max_dense_bytes=budget)
        assert torch.equal(x, b)
        passed("inclusive-budget-boundary", budget=budget)
        # Independent directional finite differences on real and imaginary value/RHS parts.
        for dtype in (torch.float64, torch.complex128):
            d = torch.tensor([[3, 0.2], [0.4, 2]], dtype=dtype)
            if dtype.is_complex:
                d[0, 1] += 0.3j
            csr = d.to_sparse_csr()
            v = csr.values().cuda().requires_grad_()
            b = torch.tensor([1, 2], dtype=dtype, device="cuda", requires_grad=True)
            crow, col = csr.crow_indices().cuda(), csr.col_indices().cuda()
            x = sparse.solve_csr(crow, col, v, b)
            gv, gb = torch.autograd.grad(x.abs().square().sum(), (v, b))
            errors = []
            for kind in ("value", "rhs"):
                for imaginary in (False, True) if dtype.is_complex else (False,):
                    delta = torch.zeros(4 if kind == "value" else 2, dtype=dtype)
                    delta[1] = 1j if imaginary else 1
                    va, vb = v.detach().cpu(), b.detach().cpu()

                    def oracle(sign):
                        aa = (
                            (va + sign * 1e-5 * delta).reshape(2, 2)
                            if kind == "value"
                            else va.reshape(2, 2)
                        )
                        bb = vb + sign * 1e-5 * delta if kind == "rhs" else vb
                        return torch.linalg.solve(aa, bb).abs().square().sum()

                    fd = (oracle(1) - oracle(-1)) / (2e-5)
                    g = (gv if kind == "value" else gb).detach().cpu()[1]
                    expected = g.imag if imaginary else g.real
                    errors.append(float(abs(fd - expected)))
                    assert abs(float(fd - expected)) < 1e-8
            passed(f"finite-differences-{dtype}", max_error=max(errors))
        # Fixed sparse structure, controlled input failures. Budget failures must
        # happen before dense allocation/solve, even for an empty sparse matrix.
        base = torch.tensor([[3.0, 1.0], [1.0, 2.0]], device="cuda").to_sparse_csr()
        crow, col, val = base.crow_indices(), base.col_indices(), base.values()
        b = torch.ones(2, device="cuda")

        def reject(name, call, before_dense=False):
            with (
                patch(
                    "torch.linalg.solve_ex", side_effect=AssertionError("entered solve")
                )
                if before_dense
                else __import__("contextlib").nullcontext()
            ):
                try:
                    call()
                except (ValueError, TypeError, RuntimeError) as error:
                    assert not isinstance(error, AssertionError)
                    passed("refusal-" + name, error=str(error))
                else:
                    raise AssertionError(name + " unexpectedly accepted")

        refusals = {
            "cpu": lambda: sparse.spsolve(base.cpu(), b.cpu()),
            "coo": lambda: sparse.spsolve(base.to_sparse_coo(), b),
            "dense": lambda: sparse.spsolve(base.to_dense(), b),
            "rectangular": lambda: sparse.spsolve(
                torch.sparse_csr_tensor(crow, col, val, size=(2, 3)), b
            ),
            "b-shape": lambda: sparse.spsolve(base, b[:1]),
            "b-dim": lambda: sparse.spsolve(base, b.reshape(2, 1, 1)),
            "b-empty": lambda: sparse.spsolve(base, b.reshape(2, 1)[:, :0]),
            "b-65rhs": lambda: sparse.spsolve(base, b[:, None].expand(2, 65)),
            "bf16": lambda: sparse.spsolve(
                base.to(torch.bfloat16), b.to(torch.bfloat16)
            ),
            "dtype-mismatch": lambda: sparse.spsolve(base, b.double()),
            "budget-bool": lambda: sparse.spsolve(base, b, max_dense_bytes=True),
            "budget-zero": lambda: sparse.spsolve(base, b, max_dense_bytes=0),
            "budget-raise": lambda: sparse.spsolve(base, b, max_dense_bytes=67108865),
            "budget-small": lambda: sparse.spsolve(base, b, max_dense_bytes=1),
            "too-large": lambda: sparse.solve_csr(
                torch.zeros(1026, device="cuda", dtype=torch.int64),
                col[:0],
                val[:0],
                torch.zeros(1025, device="cuda"),
            ),
            "zero-n": lambda: sparse.solve_csr(crow[:1], col[:0], val[:0], b[:0]),
            "index-dtype": lambda: sparse.solve_csr(crow.int(), col, val, b),
            "index-count": lambda: sparse.solve_csr(crow, col[:1], val, b),
            "index-range": lambda: sparse.solve_csr(crow, col + 2, val, b),
            "index-duplicate": lambda: sparse.solve_csr(
                crow, torch.zeros_like(col), val, b
            ),
            "index-unsorted": lambda: sparse.solve_csr(crow, col.flip(0), val, b),
            "crow-invalid": lambda: sparse.solve_csr(crow + 1, col, val, b),
            "nan-values": lambda: sparse.solve_csr(crow, col, val * float("nan"), b),
            "inf-rhs": lambda: sparse.spsolve(base, b * float("inf")),
        }
        for name, call in refusals.items():
            reject(name, call, True)
        for dtype in (torch.float32, torch.float64, torch.complex64, torch.complex128):
            singular = torch.tensor(
                [[1, 2], [2, 4]], device="cuda", dtype=dtype
            ).to_sparse_csr()
            reject(
                "singular-" + str(dtype), lambda: sparse.spsolve(singular, b.to(dtype))
            )
            empty = torch.sparse_csr_tensor(
                torch.zeros(3, device="cuda", dtype=torch.int64),
                col[:0],
                val[:0].to(dtype),
                size=(2, 2),
            )
            reject("empty-" + str(dtype), lambda: sparse.spsolve(empty, b.to(dtype)))
        original = torch.linalg.solve_ex

        def corrupt(*a, **kw):
            x, info = original(*a, **kw)
            return x + 1, info

        with patch("torch.linalg.solve_ex", corrupt):
            reject("residual", lambda: sparse.spsolve(base, b))

        def nonfinite(*a, **kw):
            x, info = original(*a, **kw)
            return x * float("inf"), info

        with patch("torch.linalg.solve_ex", nonfinite):
            reject("nonfinite-output", lambda: sparse.spsolve(base, b))
        # Native public API must retain its rejection; no global replacement.
        reject("native-api-unchanged", lambda: torch.sparse.spsolve(base, b))
        # Fresh trace of forward and both first derivatives; real HIP kernels.
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            fixture(torch.float32, torch.int64, 17, 3, "step", False)
            torch.cuda.synchronize()
        trace = args.output.parent / "sparse-b021-trace.json"
        prof.export_chrome_trace(str(trace))
        events = json.loads(trace.read_text())["traceEvents"]
        kernels = [e["name"] for e in events if e.get("cat") == "kernel"]
        solver = sorted(
            {
                k
                for k in kernels
                if any(
                    token in k.lower()
                    for token in (
                        "getrf",
                        "getrs",
                        "trsm",
                        "laswp",
                        "rocblas",
                        "gemm",
                        "ger_",
                        "scal_",
                    )
                )
            }
        )
        assert kernels and solver, kernels
        passed(
            "gpu-forward-backward-trace",
            gpu_kernel_count=len(kernels),
            solver_kernel_names=solver,
        )
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        persist()


if __name__ == "__main__":
    main()
