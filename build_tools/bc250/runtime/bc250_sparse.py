# SPDX-License-Identifier: MIT
"""Opt-in small CSR systems via GPU dense LU; not a scalable sparse solver.

Only canonical, unbatched square CSR matrices with FP32/FP64/C64/C128 values
are accepted. Differentiable scatter densification and torch.linalg.solve_ex run on the
input GPU; stored-value and RHS first derivatives use an explicit adjoint solve/product,
with AMP disabled in both directions. Higher derivatives are unsupported.
Metadata, finite, info and residual checks synchronize the host. No CPU solve,
global dispatch changes, out/right solves, factor reuse or graph capture.
"""

import torch
from torch.autograd.function import once_differentiable

__version__ = "0.1.0"
__all__ = ["spsolve", "solve_csr"]

_MAX_N = 1024
_MAX_RHS = 64
_MAX_DENSE_BYTES = 64 * 1024**2
_DTYPES = (torch.float32, torch.float64, torch.complex64, torch.complex128)


class _DenseSolve(torch.autograd.Function):
    """Keep the first adjoint solve/product in the original precision under AMP."""

    @staticmethod
    def forward(ctx, a, b):
        with torch.autocast("cuda", enabled=False):
            x, info = torch.linalg.solve_ex(a, b, check_errors=False)
        ctx.save_for_backward(a, x)
        ctx.mark_non_differentiable(info)
        return x, info

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_x, grad_info):
        a, x = ctx.saved_tensors
        with torch.autocast("cuda", enabled=False):
            adjoint = torch.linalg.solve(a.mH, grad_x)
            if x.ndim == 1:
                grad_a = -adjoint[:, None] @ x.conj()[None, :]
            else:
                grad_a = -adjoint @ x.mH
        return grad_a if ctx.needs_input_grad[0] else None, (
            adjoint if ctx.needs_input_grad[1] else None
        )


def spsolve(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    max_dense_bytes: int = _MAX_DENSE_BYTES,
) -> torch.Tensor:
    """Solve AX=B explicitly using bounded-size GPU dense LU.

    Accepts n=1..1024 and vector or 1..64-column RHS. The configurable budget
    may only LOWER the 64MiB ceiling. Before densification it bounds an estimate
    of major dense intermediates: element_size*(8*n*n + 6*n*nrhs). This is not
    an allocator/rocSOLVER workspace or total process memory guarantee. O(n^2)
    storage/O(n^3) work remain; large sparse problems are refused.

    Canonical CSR requires sorted unique columns per row and valid zero-based
    indices. Malformed CSR, nonfinite inputs, exact singular LU info, nonfinite
    solution or normalized residual above 64*eps raise. The residual check does
    NOT establish good conditioning or a forward error bound. First gradients
    are defined for fixed sparsity, including explicitly stored zero values.
    """
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("a and b must be Torch tensors")
    if a.layout != torch.sparse_csr or a.ndim != 2:
        raise ValueError("a must be an unbatched 2D CSR tensor")
    if a.shape[0] != a.shape[1] or not 1 <= a.shape[0] <= _MAX_N:
        raise ValueError("a must be square with 1..1024 rows")
    n = a.shape[0]
    if b.layout != torch.strided or b.ndim not in (1, 2) or b.shape[0] != n:
        raise ValueError("b must be a strided vector or matrix with n rows")
    nrhs = 1 if b.ndim == 1 else b.shape[1]
    if not 1 <= nrhs <= _MAX_RHS:
        raise ValueError("b must have 1..64 right hand sides")
    if a.dtype not in _DTYPES or b.dtype != a.dtype:
        raise TypeError("a and b must have the same FP32/FP64/C64/C128 dtype")
    if a.device.type != "cuda" or b.device != a.device:
        raise ValueError("a and b must be on the same GPU")
    if type(max_dense_bytes) is not int or not 1 <= max_dense_bytes <= _MAX_DENSE_BYTES:
        raise ValueError("max_dense_bytes must be an integer in 1..67108864")
    estimate = a.element_size() * (8 * n * n + 6 * n * nrhs)
    if estimate > max_dense_bytes:
        raise ValueError(
            f"dense storage estimate {estimate} exceeds budget {max_dense_bytes}"
        )
    if a.is_complex() and a.requires_grad:
        raise ValueError(
            "complex CSR tensor autograd is unsupported; use solve_csr with stored values"
        )
    if a.is_complex():
        # A complex CSR values view cannot be used by some real-view consumers
        # in this Torch revision (the CSR base has no strides). No A gradient
        # is admitted above; materialize detached values, retaining RHS autograd.
        with torch.no_grad():
            values = a.values().clone()
    else:
        values = a.values()
    return solve_csr(
        a.crow_indices(),
        a.col_indices(),
        values,
        b,
        max_dense_bytes=max_dense_bytes,
    )


def solve_csr(
    crow: torch.Tensor,
    col: torch.Tensor,
    values: torch.Tensor,
    b: torch.Tensor,
    *,
    max_dense_bytes: int = _MAX_DENSE_BYTES,
) -> torch.Tensor:
    """Solve square CSR parts, retaining real/complex stored-value gradients.

    n is inferred from crow.numel()-1. Same admission/storage/residual policy as
    spsolve; this avoids Torch's complex sparse-constructor autograd restriction.
    Only fixed sorted unique sparsity is differentiable; indices have no gradient.
    """
    if not all(isinstance(t, torch.Tensor) for t in (crow, col, values, b)):
        raise TypeError("CSR parts and b must be Torch tensors")
    if any(t.layout != torch.strided or t.ndim != 1 for t in (crow, col, values)):
        raise ValueError("CSR parts must be strided 1D tensors")
    n = crow.numel() - 1
    if not 1 <= n <= _MAX_N:
        raise ValueError("CSR must have 1..1024 rows")
    if b.layout != torch.strided or b.ndim not in (1, 2) or b.shape[0] != n:
        raise ValueError("b must be a strided vector or matrix with n rows")
    nrhs = 1 if b.ndim == 1 else b.shape[1]
    if not 1 <= nrhs <= _MAX_RHS:
        raise ValueError("b must have 1..64 right hand sides")
    if values.dtype not in _DTYPES or b.dtype != values.dtype:
        raise TypeError("values and b must have the same FP32/FP64/C64/C128 dtype")
    if values.device.type != "cuda" or any(
        t.device != values.device for t in (crow, col, b)
    ):
        raise ValueError("CSR parts and b must be on the same GPU")
    if crow.dtype not in (torch.int32, torch.int64) or col.dtype != crow.dtype:
        raise TypeError("CSR indices must have matching int32/int64 dtype")
    if type(max_dense_bytes) is not int or not 1 <= max_dense_bytes <= _MAX_DENSE_BYTES:
        raise ValueError("max_dense_bytes must be an integer in 1..67108864")
    estimate = values.element_size() * (8 * n * n + 6 * n * nrhs)
    if estimate > max_dense_bytes:
        raise ValueError(
            f"dense storage estimate {estimate} exceeds budget {max_dense_bytes}"
        )
    if values.numel() > n * n or col.numel() != values.numel():
        raise ValueError(
            "CSR must have matching scalar columns/values and at most n*n entries"
        )
    with torch.no_grad():
        nnz = values.numel()
        # Do not use the native GPU CSR invariant checker here: invalid inputs
        # raise device assertions on this stack and poison the process queue.
        # These predicates never dereference a CSR index before it is checked.
        if not bool((crow[0] == 0) & (crow[-1] == nnz)):
            raise ValueError("CSR row pointers must start at zero and end at nnz")
        if not bool(((crow >= 0) & (crow <= nnz)).all()) or not bool(
            (crow[1:] >= crow[:-1]).all()
        ):
            raise ValueError("CSR row pointers must be monotonic within 0..nnz")
        if not bool(((col >= 0) & (col < n)).all()):
            raise ValueError("CSR column indices must be within 0..n-1")
        rows = torch.repeat_interleave(
            torch.arange(n, device=values.device),
            crow[1:] - crow[:-1],
            output_size=nnz,
        )
        if bool(((rows[1:] == rows[:-1]) & (col[1:] <= col[:-1])).any()):
            raise ValueError("CSR columns must be sorted and unique within each row")
        if not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(b).all()):
            raise ValueError("values and b must contain only finite values")
    # Disable caller AMP: conversion/solve/residual policies are explicit here.
    with torch.autocast("cuda", enabled=False):
        dense = torch.zeros((n, n), dtype=values.dtype, device=values.device).index_put(
            (rows, col.long()),
            values,
        )
        x, info = _DenseSolve.apply(dense, b)
        code = int(info)
        if code > 0:
            raise torch.linalg.LinAlgError(f"singular matrix: LU zero pivot {code}")
        if code < 0:
            raise RuntimeError(f"native solve_ex failed with info={code}")
        with torch.no_grad():
            if not bool(torch.isfinite(x).all()):
                raise torch.linalg.LinAlgError("nonfinite solution")
            dtype = torch.complex128 if dense.is_complex() else torch.float64
            da, dx, db = dense.to(dtype), x.to(dtype), b.to(dtype)
            if dx.ndim == 1:
                dx, db = dx[:, None], db[:, None]
            numerator = (da @ dx - db).abs().amax(dim=0)
            denominator = da.abs().sum(dim=1).amax() * dx.abs().amax(
                dim=0
            ) + db.abs().amax(dim=0)
            if not bool(torch.isfinite(denominator).all()):
                raise torch.linalg.LinAlgError("nonfinite residual scale")
            residual = numerator / denominator.clamp_min(
                torch.finfo(torch.float64).tiny
            )
            bound = 64 * torch.finfo(values.real.dtype).eps
            if not bool(torch.isfinite(residual).all()) or bool(
                (residual > bound).any()
            ):
                raise torch.linalg.LinAlgError(
                    "normalized solve residual exceeds 64*eps"
                )
    return x
