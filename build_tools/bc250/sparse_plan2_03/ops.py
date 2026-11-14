# SPDX-License-Identifier: MIT
"""Native hipSPARSE ILU(0)/IC(0) factorization + triangular solve for gfx1013.

Private-handle ctypes path to libhipsparse (csrilu02/csric02/csrsv2), reusing the
N2.13 native-validation pattern. The pinned general Torch sparse-solve API rejects
ROCm, so this is an explicit opt-in native route, not a dispatch/default change.
Strict ILU(0)/IC(0): numeric boost disabled, zero pivot reported (never silently
perturbed). Device memory comes from torch tensors (data_ptr); no native rebuild.
"""

import ctypes

import torch

from . import core

STATUS_SUCCESS = 0
INDEX_BASE_ZERO = 0
DIAG_TYPE_NON_UNIT = 0
DIAG_TYPE_UNIT = 1
OP_NON_TRANSPOSE = 0
OP_TRANSPOSE = 1
SOLVE_POLICY_USE_LEVEL = 1
MATRIX_TYPE_TRIANGULAR = 3
FILL_MODE_LOWER = 0
FILL_MODE_UPPER = 1

_PREFIX = {torch.float32: "S", torch.float64: "D"}


def _device_guard(device) -> None:
    if device.type != "cuda" or not torch.version.hip:
        raise ValueError("native sparse factorization requires gfx1013 HIP tensors")
    if not torch.cuda.get_device_properties(device).gcnArchName.startswith("gfx1013:"):
        raise ValueError("native sparse factorization is qualified only on gfx1013")


class HipSparse:
    """Minimal ctypes binding to the classic hipSPARSE factorization/solve API."""

    def __init__(self, library="libhipsparse.so"):
        self.lib = ctypes.CDLL(library)
        self.P = ctypes.c_void_p
        self.I = ctypes.c_int
        self.PP = ctypes.POINTER(self.P)
        self.PI = ctypes.POINTER(self.I)
        self.PD = ctypes.POINTER(ctypes.c_double)
        self._bind()

    def _fn(self, name, argtypes):
        f = getattr(self.lib, name)
        f.argtypes = argtypes
        f.restype = self.I
        return f

    def _bind(self):
        P, I, PP, PI, PD = self.P, self.I, self.PP, self.PI, self.PD
        # Classic hipSPARSE naming: hipsparseCreate/Destroy (not *Handle).
        self.create_handle = self._fn("hipsparseCreate", [PP])
        self.destroy_handle = self._fn("hipsparseDestroy", [P])
        self.create_descr = self._fn("hipsparseCreateMatDescr", [PP])
        self.destroy_descr = self._fn("hipsparseDestroyMatDescr", [P])
        self.set_index_base = self._fn("hipsparseSetMatIndexBase", [P, I])
        self.set_diag_type = self._fn("hipsparseSetMatDiagType", [P, I])
        self.set_mat_type = self._fn("hipsparseSetMatType", [P, I])
        self.set_fill_mode = self._fn("hipsparseSetMatFillMode", [P, I])
        for info in ("Csrilu02", "Csric02", "Csrsv2"):
            setattr(
                self, f"create_{info}", self._fn(f"hipsparseCreate{info}Info", [PP])
            )
            setattr(
                self, f"destroy_{info}", self._fn(f"hipsparseDestroy{info}Info", [P])
            )
        # factorization: handle, m, nnz, descr, val, rowptr, colind, info, [policy, buffer]
        fac_args = [P, I, I, P, P, P, P, P]
        for p in ("S", "D"):
            setattr(
                self,
                f"{p}_csrilu02_bufferSize",
                self._fn(f"hipsparse{p}csrilu02_bufferSize", fac_args + [PI]),
            )
            setattr(
                self,
                f"{p}_csrilu02_analysis",
                self._fn(f"hipsparse{p}csrilu02_analysis", fac_args + [I, P]),
            )
            setattr(
                self,
                f"{p}_csrilu02",
                self._fn(f"hipsparse{p}csrilu02", fac_args + [I, P]),
            )
            setattr(
                self,
                f"{p}_csrilu02_numericBoost",
                self._fn(f"hipsparse{p}csrilu02_numericBoost", [P, P, I, PD, P]),
            )
            setattr(
                self,
                f"{p}_csric02_bufferSize",
                self._fn(f"hipsparse{p}csric02_bufferSize", fac_args + [PI]),
            )
            setattr(
                self,
                f"{p}_csric02_analysis",
                self._fn(f"hipsparse{p}csric02_analysis", fac_args + [I, P]),
            )
            setattr(
                self,
                f"{p}_csric02",
                self._fn(f"hipsparse{p}csric02", fac_args + [I, P]),
            )
            sv_args = [P, I, I, I, P, P, P, P, P]
            setattr(
                self,
                f"{p}_csrsv2_bufferSize",
                self._fn(f"hipsparse{p}csrsv2_bufferSize", sv_args + [PI]),
            )
            setattr(
                self,
                f"{p}_csrsv2_analysis",
                self._fn(f"hipsparse{p}csrsv2_analysis", sv_args + [I, P]),
            )
            setattr(
                self,
                f"{p}_csrsv2_solve",
                self._fn(
                    f"hipsparse{p}csrsv2_solve",
                    [P, I, I, I, P, P, P, P, P, P, P, P, I, P],
                ),
            )
        self.xcsrilu02_zero_pivot = self._fn("hipsparseXcsrilu02_zeroPivot", [P, P, PI])
        self.xcsric02_zero_pivot = self._fn("hipsparseXcsric02_zeroPivot", [P, P, PI])

    def _check(self, status, what):
        if status != STATUS_SUCCESS:
            raise RuntimeError(f"hipSPARSE {what} failed with status {status}")


def _csr_tensors(row_ptr, col_ind, values, n, device, dtype):
    rp = torch.tensor(row_ptr, device=device, dtype=torch.int32)
    ci = torch.tensor(col_ind, device=device, dtype=torch.int32)
    val = torch.tensor(values, device=device, dtype=dtype)
    return rp, ci, val


def _validate_csr(row_ptr, col_ind, values, n):
    if n < 1 or len(row_ptr) != n + 1:
        raise ValueError("CSR must be square nonempty with n+1 row pointers")
    if row_ptr[0] != 0 or row_ptr[-1] != len(values) or len(col_ind) != len(values):
        raise ValueError("CSR metadata mismatch")
    if any(row_ptr[i + 1] < row_ptr[i] for i in range(n)):
        raise ValueError("CSR row pointers must be nondecreasing")
    if any(not (0 <= c < n) for c in col_ind):
        raise ValueError("CSR column index out of range")
    if not all(torch.isfinite(torch.tensor(values, dtype=torch.float64)).tolist()):
        raise ValueError("CSR values must be finite")


def ilu0_factor(row_ptr, col_ind, values, n, *, device, dtype=torch.float32):
    """Native ILU(0) (csrilu02). Returns (factored_values list, zero_pivot_row).

    Strict no-fill ILU(0): numeric boost disabled. zero_pivot_row is -1 when
    nonsingular. The input values are not modified.
    """
    _device_guard(device)
    if dtype not in _PREFIX:
        raise ValueError("ILU(0) supports float32/float64")
    _validate_csr(row_ptr, col_ind, values, n)
    p = _PREFIX[dtype]
    hs = HipSparse()
    handle, descr, info = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    hs._check(hs.create_handle(ctypes.byref(handle)), "create_handle")
    hs._check(hs.create_descr(ctypes.byref(descr)), "create_descr")
    hs._check(hs.set_index_base(descr, INDEX_BASE_ZERO), "set_index_base")
    hs._check(hs.create_Csrilu02(ctypes.byref(info)), "create_info")
    try:
        rp, ci, val = _csr_tensors(row_ptr, col_ind, values, n, device, dtype)
        nnz = len(values)
        # Disable numeric boost for strict ILU(0).
        tol = ctypes.c_double(0.0)
        boost = ctypes.c_float(0.0) if dtype == torch.float32 else ctypes.c_double(0.0)
        hs._check(
            getattr(hs, f"{p}_csrilu02_numericBoost")(
                handle, info, 0, ctypes.byref(tol), ctypes.byref(boost)
            ),
            "numeric_boost",
        )
        size = ctypes.c_int(0)
        hs._check(
            getattr(hs, f"{p}_csrilu02_bufferSize")(
                handle,
                n,
                nnz,
                descr,
                val.data_ptr(),
                rp.data_ptr(),
                ci.data_ptr(),
                info,
                ctypes.byref(size),
            ),
            "buffer_size",
        )
        buf = torch.empty(max(size.value, 1), device=device, dtype=torch.uint8)
        hs._check(
            getattr(hs, f"{p}_csrilu02_analysis")(
                handle,
                n,
                nnz,
                descr,
                val.data_ptr(),
                rp.data_ptr(),
                ci.data_ptr(),
                info,
                SOLVE_POLICY_USE_LEVEL,
                buf.data_ptr(),
            ),
            "analysis",
        )
        hs._check(
            getattr(hs, f"{p}_csrilu02")(
                handle,
                n,
                nnz,
                descr,
                val.data_ptr(),
                rp.data_ptr(),
                ci.data_ptr(),
                info,
                SOLVE_POLICY_USE_LEVEL,
                buf.data_ptr(),
            ),
            "factor",
        )
        torch.cuda.synchronize(device)
        pos = ctypes.c_int(-1)
        hs.xcsrilu02_zero_pivot(handle, info, ctypes.byref(pos))
        return val.detach().cpu().tolist(), pos.value
    finally:
        hs.destroy_Csrilu02(info)
        hs.destroy_descr(descr)
        hs.destroy_handle(handle)


def ic0_factor(row_ptr, col_ind, values, n, *, device, dtype=torch.float32):
    """Native IC(0) (csric02) on the stored LOWER triangle. Returns (factored, pivot)."""
    _device_guard(device)
    if dtype not in _PREFIX:
        raise ValueError("IC(0) supports float32/float64")
    _validate_csr(row_ptr, col_ind, values, n)
    p = _PREFIX[dtype]
    hs = HipSparse()
    handle, descr, info = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    hs._check(hs.create_handle(ctypes.byref(handle)), "create_handle")
    hs._check(hs.create_descr(ctypes.byref(descr)), "create_descr")
    hs._check(hs.set_index_base(descr, INDEX_BASE_ZERO), "set_index_base")
    hs._check(hs.create_Csric02(ctypes.byref(info)), "create_info")
    try:
        rp, ci, val = _csr_tensors(row_ptr, col_ind, values, n, device, dtype)
        nnz = len(values)
        size = ctypes.c_int(0)
        hs._check(
            getattr(hs, f"{p}_csric02_bufferSize")(
                handle,
                n,
                nnz,
                descr,
                val.data_ptr(),
                rp.data_ptr(),
                ci.data_ptr(),
                info,
                ctypes.byref(size),
            ),
            "buffer_size",
        )
        buf = torch.empty(max(size.value, 1), device=device, dtype=torch.uint8)
        hs._check(
            getattr(hs, f"{p}_csric02_analysis")(
                handle,
                n,
                nnz,
                descr,
                val.data_ptr(),
                rp.data_ptr(),
                ci.data_ptr(),
                info,
                SOLVE_POLICY_USE_LEVEL,
                buf.data_ptr(),
            ),
            "analysis",
        )
        hs._check(
            getattr(hs, f"{p}_csric02")(
                handle,
                n,
                nnz,
                descr,
                val.data_ptr(),
                rp.data_ptr(),
                ci.data_ptr(),
                info,
                SOLVE_POLICY_USE_LEVEL,
                buf.data_ptr(),
            ),
            "factor",
        )
        torch.cuda.synchronize(device)
        pos = ctypes.c_int(-1)
        hs.xcsric02_zero_pivot(handle, info, ctypes.byref(pos))
        return val.detach().cpu().tolist(), pos.value
    finally:
        hs.destroy_Csric02(info)
        hs.destroy_descr(descr)
        hs.destroy_handle(handle)


def split_lu_csr(row_ptr, col_ind, factored, n):
    """Split ILU(0) factors into L (unit lower) and U (upper) CSR lists."""
    L_rp, L_ci, L_val = [0], [], []
    U_rp, U_ci, U_val = [0], [], []
    for i in range(n):
        for p in range(row_ptr[i], row_ptr[i + 1]):
            j = col_ind[p]
            if j < i:
                L_ci.append(j)
                L_val.append(factored[p])
        L_ci.append(i)
        L_val.append(1.0)
        L_rp.append(len(L_val))
        for p in range(row_ptr[i], row_ptr[i + 1]):
            j = col_ind[p]
            if j >= i:
                U_ci.append(j)
                U_val.append(factored[p])
        U_rp.append(len(U_val))
    return (L_rp, L_ci, L_val), (U_rp, U_ci, U_val)


def _torch_matvec(rp, ci, val, x, n):
    row_idx = torch.repeat_interleave(
        torch.arange(n, device=x.device), (rp[1:] - rp[:-1]).long()
    )
    return torch.zeros(n, device=x.device, dtype=x.dtype).index_add_(
        0, row_idx, val * x[ci]
    )


class TriangularSolver:
    """Reusable native csrsv2 triangular solve (analysis done once)."""

    def __init__(
        self,
        row_ptr,
        col_ind,
        values,
        n,
        *,
        fill_mode,
        diag_type,
        device,
        dtype,
        trans=OP_NON_TRANSPOSE,
    ):
        _device_guard(device)
        if dtype not in _PREFIX:
            raise ValueError("csrsv2 supports float32/float64")
        self.device, self.dtype, self.n = device, dtype, n
        # csrsv2 analysis is trans-specific: bind it to the solve operation.
        self.trans = trans
        self.p = _PREFIX[dtype]
        self.hs = HipSparse()
        self.handle = ctypes.c_void_p()
        self.descr = ctypes.c_void_p()
        self.info = ctypes.c_void_p()
        hs = self.hs
        hs._check(hs.create_handle(ctypes.byref(self.handle)), "create_handle")
        hs._check(hs.create_descr(ctypes.byref(self.descr)), "create_descr")
        hs._check(hs.set_index_base(self.descr, INDEX_BASE_ZERO), "index_base")
        hs._check(hs.set_mat_type(self.descr, MATRIX_TYPE_TRIANGULAR), "mat_type")
        hs._check(hs.set_fill_mode(self.descr, fill_mode), "fill_mode")
        hs._check(hs.set_diag_type(self.descr, diag_type), "diag_type")
        hs._check(hs.create_Csrsv2(ctypes.byref(self.info)), "create_info")
        self.rp, self.ci, self.val = _csr_tensors(
            row_ptr, col_ind, values, n, device, dtype
        )
        self.alpha = torch.ones(1, device=device, dtype=dtype)
        nnz = len(values)
        size = ctypes.c_int(0)
        hs._check(
            getattr(hs, f"{self.p}_csrsv2_bufferSize")(
                self.handle,
                self.trans,
                n,
                nnz,
                self.descr,
                self.val.data_ptr(),
                self.rp.data_ptr(),
                self.ci.data_ptr(),
                self.info,
                ctypes.byref(size),
            ),
            "buffer_size",
        )
        self.buf = torch.empty(max(size.value, 1), device=device, dtype=torch.uint8)
        hs._check(
            getattr(hs, f"{self.p}_csrsv2_analysis")(
                self.handle,
                self.trans,
                n,
                nnz,
                self.descr,
                self.val.data_ptr(),
                self.rp.data_ptr(),
                self.ci.data_ptr(),
                self.info,
                SOLVE_POLICY_USE_LEVEL,
                self.buf.data_ptr(),
            ),
            "analysis",
        )

    def solve(self, rhs):
        x = torch.as_tensor(rhs, device=self.device, dtype=self.dtype).contiguous()
        y = torch.empty(self.n, device=self.device, dtype=self.dtype)
        self.hs._check(
            getattr(self.hs, f"{self.p}_csrsv2_solve")(
                self.handle,
                self.trans,
                self.n,
                len(self.val),
                self.alpha.data_ptr(),
                self.descr,
                self.val.data_ptr(),
                self.rp.data_ptr(),
                self.ci.data_ptr(),
                self.info,
                x.data_ptr(),
                y.data_ptr(),
                SOLVE_POLICY_USE_LEVEL,
                self.buf.data_ptr(),
            ),
            "solve",
        )
        return y

    def close(self):
        self.hs.destroy_Csrsv2(self.info)
        self.hs.destroy_descr(self.descr)
        self.hs.destroy_handle(self.handle)


def ilu_preconditioned_solve(
    row_ptr, col_ind, values, n, rhs, *, iters, device, dtype=torch.float32
):
    """ILU(0)-preconditioned Richardson solve of A x = b (native factor + csrsv2).

    Returns (x list, relative_residual, zero_pivot). Non-convergence is reported
    via the residual, never hidden.
    """
    factored, pivot = ilu0_factor(
        row_ptr, col_ind, values, n, device=device, dtype=dtype
    )
    if pivot >= 0:
        raise ValueError(f"ILU(0) zero pivot at row {pivot}; cannot precondition")
    (L_rp, L_ci, L_val), (U_rp, U_ci, U_val) = split_lu_csr(
        row_ptr, col_ind, factored, n
    )
    L = TriangularSolver(
        L_rp,
        L_ci,
        L_val,
        n,
        fill_mode=FILL_MODE_LOWER,
        diag_type=DIAG_TYPE_UNIT,
        device=device,
        dtype=dtype,
    )
    U = TriangularSolver(
        U_rp,
        U_ci,
        U_val,
        n,
        fill_mode=FILL_MODE_UPPER,
        diag_type=DIAG_TYPE_NON_UNIT,
        device=device,
        dtype=dtype,
    )
    try:
        rp, ci, val = _csr_tensors(row_ptr, col_ind, values, n, device, dtype)
        b = torch.as_tensor(rhs, device=device, dtype=dtype)
        x = torch.zeros(n, device=device, dtype=dtype)
        for _ in range(iters):
            r = b - _torch_matvec(rp, ci, val, x, n)
            x = x + U.solve(L.solve(r))
        torch.cuda.synchronize(device)
        resid = (b - _torch_matvec(rp, ci, val, x, n)).norm() / b.norm().clamp_min(
            1e-300
        )
        return x.detach().cpu().tolist(), resid.item(), pivot
    finally:
        L.close()
        U.close()


def ilu_adjoint_solve(
    row_ptr, col_ind, values, n, rhs, *, iters, device, dtype=torch.float32
):
    """ILU(0)-preconditioned Richardson solve of A^T g = rhs (for the RHS gradient)."""
    factored, pivot = ilu0_factor(
        row_ptr, col_ind, values, n, device=device, dtype=dtype
    )
    if pivot >= 0:
        raise ValueError(f"ILU(0) zero pivot at row {pivot}; cannot precondition")
    (L_rp, L_ci, L_val), (U_rp, U_ci, U_val) = split_lu_csr(
        row_ptr, col_ind, factored, n
    )
    # Adjoint preconditioner M^{-T} = L^{-T} U^{-T}: analyze both for TRANSPOSE.
    L = TriangularSolver(
        L_rp,
        L_ci,
        L_val,
        n,
        fill_mode=FILL_MODE_LOWER,
        diag_type=DIAG_TYPE_UNIT,
        device=device,
        dtype=dtype,
        trans=OP_TRANSPOSE,
    )
    U = TriangularSolver(
        U_rp,
        U_ci,
        U_val,
        n,
        fill_mode=FILL_MODE_UPPER,
        diag_type=DIAG_TYPE_NON_UNIT,
        device=device,
        dtype=dtype,
        trans=OP_TRANSPOSE,
    )
    try:
        # A^T matvec via the transpose CSR pattern.
        t_rp, t_ci, t_val = _transpose_csr(row_ptr, col_ind, values, n)
        trp, tci, tval = _csr_tensors(t_rp, t_ci, t_val, n, device, dtype)
        b = torch.as_tensor(rhs, device=device, dtype=dtype)
        g = torch.zeros(n, device=device, dtype=dtype)
        for _ in range(iters):
            r = b - _torch_matvec(trp, tci, tval, g, n)
            g = g + L.solve(U.solve(r))
        torch.cuda.synchronize(device)
        resid = (b - _torch_matvec(trp, tci, tval, g, n)).norm() / b.norm().clamp_min(
            1e-300
        )
        return g.detach().cpu().tolist(), resid.item(), pivot
    finally:
        L.close()
        U.close()


def _transpose_csr(row_ptr, col_ind, values, n):
    counts = [0] * n
    for c in col_ind:
        counts[c] += 1
    t_rp = [0]
    for c in counts:
        t_rp.append(t_rp[-1] + c)
    t_ci = [0] * len(values)
    t_val = [0.0] * len(values)
    cursor = list(t_rp[:-1])
    for i in range(n):
        for p in range(row_ptr[i], row_ptr[i + 1]):
            j = col_ind[p]
            dest = cursor[j]
            t_ci[dest] = i
            t_val[dest] = values[p]
            cursor[j] += 1
    return t_rp, t_ci, t_val
