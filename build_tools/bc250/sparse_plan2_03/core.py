# SPDX-License-Identifier: MIT
"""Pure CPU ILU(0)/IC(0) reference and CSR helpers (no torch).

Independent parity anchor for the native hipSPARSE csrilu02/csric02 factorization.
The references use the classic no-fill IKJ (ILU(0)) and row-oriented IC(0)
algorithms on the SAME sparsity pattern as the input (zero fill dropped), with
strict zero-pivot detection and no numeric boost/perturbation. Dependency-free so
the contract is testable on the workstation; the GPU path lives in ops.py.
"""

import math


def dense_to_csr(dense):
    """Dense n×n list -> (row_ptr, col_ind, values, n); keeps explicit zeros' pattern
    only for nonzeros (structural zeros are dropped)."""
    n = len(dense)
    row_ptr, col_ind, values = [0], [], []
    for i in range(n):
        for j in range(n):
            if dense[i][j] != 0.0:
                col_ind.append(j)
                values.append(float(dense[i][j]))
        row_ptr.append(len(values))
    return row_ptr, col_ind, values, n


def csr_to_dense(row_ptr, col_ind, values, n):
    dense = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for p in range(row_ptr[i], row_ptr[i + 1]):
            dense[i][col_ind[p]] = values[p]
    return dense


def _row_maps(row_ptr, col_ind, n):
    maps = []
    for i in range(n):
        maps.append({col_ind[p]: p for p in range(row_ptr[i], row_ptr[i + 1])})
    return maps


def ilu0_reference(row_ptr, col_ind, values, n):
    """No-fill ILU(0) (IKJ). Returns (factored_values, zero_pivot_row).

    Below-diagonal entries hold L (unit diagonal implicit); diagonal and above
    hold U. zero_pivot_row is the first zero diagonal encountered, or -1.
    """
    a = [float(v) for v in values]
    maps = _row_maps(row_ptr, col_ind, n)
    diag = [maps[i].get(i, -1) for i in range(n)]
    for i in range(n):
        if diag[i] < 0:
            return a, i  # structural zero diagonal
    for i in range(1, n):
        for p_k in range(row_ptr[i], row_ptr[i + 1]):
            k = col_ind[p_k]
            if k >= i:
                break
            dk = diag[k]
            if a[dk] == 0.0:
                return a, k
            a[p_k] = a[p_k] / a[dk]
            aik = a[p_k]
            for p_j in range(p_k + 1, row_ptr[i + 1]):
                j = col_ind[p_j]
                if j <= k:
                    continue
                p_kj = maps[k].get(j, -1)
                if p_kj >= 0:
                    a[p_j] = a[p_j] - aik * a[p_kj]
    for i in range(n):
        if a[diag[i]] == 0.0:
            return a, i
    return a, -1


def ic0_reference(row_ptr, col_ind, values, n):
    """No-fill IC(0) on the stored LOWER triangle. Returns (factored_values, pivot).

    Diagonal holds the Cholesky diagonal; below-diagonal holds L. pivot is the
    first non-positive pivot row, or -1. Input must be the lower triangle of an
    SPD-pattern matrix.
    """
    a = [float(v) for v in values]
    maps = _row_maps(row_ptr, col_ind, n)
    diag = [maps[i].get(i, -1) for i in range(n)]
    for i in range(n):
        if diag[i] < 0:
            return a, i
        for p_j in range(row_ptr[i], diag[i]):
            j = col_ind[p_j]
            if j >= i:
                continue
            s = a[p_j]
            for p_k in range(row_ptr[i], p_j):
                k = col_ind[p_k]
                if k >= j:
                    break
                p_jk = maps[j].get(k, -1)
                if p_jk >= 0:
                    s -= a[p_k] * a[p_jk]
            if a[diag[j]] == 0.0:
                return a, j
            a[p_j] = s / a[diag[j]]
        s = a[diag[i]]
        for p_k in range(row_ptr[i], diag[i]):
            k = col_ind[p_k]
            if k < i:
                s -= a[p_k] * a[p_k]
        if s <= 0.0:
            return a, i
        a[diag[i]] = math.sqrt(s)
    return a, -1


def csr_matvec(row_ptr, col_ind, values, n, x):
    y = [0.0] * n
    for i in range(n):
        s = 0.0
        for p in range(row_ptr[i], row_ptr[i + 1]):
            s += values[p] * x[col_ind[p]]
        y[i] = s
    return y


def dense_solve(A, b):
    """Gaussian elimination with partial pivoting (small fixtures); returns x."""
    n = len(A)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-300:
            raise ValueError("singular matrix")
        M[c], M[piv] = M[piv], M[c]
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            for j in range(c, n + 1):
                M[r][j] -= f * M[c][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / M[i][i]
    return x


def split_lu(row_ptr, col_ind, factored, n):
    """Split factored CSR values into dense L (unit diag) and U for reconstruction."""
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    for i in range(n):
        L[i][i] = 1.0
        for p in range(row_ptr[i], row_ptr[i + 1]):
            j = col_ind[p]
            if j < i:
                L[i][j] = factored[p]
            else:
                U[i][j] = factored[p]
    return L, U


def matmul_dense(A, B):
    n = len(A)
    return [
        [sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)] for i in range(n)
    ]


def forward_sub_unit_lower(row_ptr, col_ind, values, n, b):
    """Solve L x = b for unit-lower-triangular CSR L (diagonal implicit 1)."""
    x = [0.0] * n
    for i in range(n):
        s = b[i]
        for p in range(row_ptr[i], row_ptr[i + 1]):
            j = col_ind[p]
            if j < i:
                s -= values[p] * x[j]
        x[i] = s
    return x


def back_sub_upper(row_ptr, col_ind, values, n, b):
    """Solve U x = b for upper-triangular CSR U (explicit diagonal)."""
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = b[i]
        diag = None
        for p in range(row_ptr[i], row_ptr[i + 1]):
            j = col_ind[p]
            if j > i:
                s -= values[p] * x[j]
            elif j == i:
                diag = values[p]
        x[i] = s / diag
    return x


def transpose_csr(row_ptr, col_ind, values, n):
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
