#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PLAN2-03 native sparse factorization GPU qualification (inside the guardian).

Native gfx1013 only. Qualifies hipSPARSE ILU(0)/IC(0) factorization against the
independent pure-Python CPU reference within the pre-declared bounds
(tmp/sparse-plan2-03/BOUNDS.md), the zero/singular-pivot info contract and the
refusal surface. Writes a JSON report; exits non-zero on any miss. No native
operator/dispatch/default/source change.
"""

import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from sparse_plan2_03 import core, ops  # noqa: E402

# Pre-declared factor parity bounds (GPU vs independent CPU reference).
FACTOR_BOUNDS = {
    torch.float32: (1e-3, 1e-5, 1e-3),
    torch.float64: (1e-9, 1e-11, 1e-8),
}

REPORT = {"checks": [], "measurements": {}, "environment": {}}


def check(name, passed, **detail):
    REPORT["checks"].append({"name": name, "passed": bool(passed), **detail})
    if not passed:
        print(f"SPARSE_FAIL {name} {detail}", flush=True)
    return passed


def rel_l2(a, b):
    num = sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
    den = sum(y * y for y in b) ** 0.5
    return num / max(den, 1e-300)


def factor_parity(a, b, dtype):
    rtol, atol, l2 = FACTOR_BOUNDS[dtype]
    ok = all(abs(x - y) <= atol + rtol * abs(y) for x, y in zip(a, b))
    rl2 = rel_l2(a, b)
    return ok and rl2 < l2, {
        "allclose": ok,
        "rel_l2": rl2,
        "max_abs": max((abs(x - y) for x, y in zip(a, b)), default=0.0),
        "l2_bound": l2,
    }


def tridiagonal(n, diag=4.0, off=1.0):
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        A[i][i] = diag
        if i > 0:
            A[i][i - 1] = off
        if i < n - 1:
            A[i][i + 1] = off
    return A


def dense_pattern(n, shift):
    return [
        [1.0 / (i + j + 1) + (shift if i == j else 0.0) for j in range(n)]
        for i in range(n)
    ]


def general_sparse(n, seed, density=0.15):
    rng = random.Random(seed)
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and rng.random() < density:
                A[i][j] = rng.uniform(-1.0, 1.0)
        A[i][i] = sum(abs(A[i][j]) for j in range(n)) + 1.0  # diagonal dominant
    return A


def environment(device):
    env = REPORT["environment"]
    env["torch"] = torch.__version__
    env["hip"] = torch.version.hip
    env["gcnArchName"] = torch.cuda.get_device_properties(device).gcnArchName
    check(
        "gfx1013-native",
        env["gcnArchName"].startswith("gfx1013:"),
        arch=env["gcnArchName"],
    )


def ilu0_parity(device):
    fixtures = {
        "tridiag64": tridiagonal(64),
        "tridiag256": tridiagonal(256),
        "dense16": dense_pattern(16, 16),
        "dense32": dense_pattern(32, 32),
        "general64": general_sparse(64, 20261114),
    }
    worst = {torch.float32: 0.0, torch.float64: 0.0}
    for dtype in (torch.float32, torch.float64):
        for name, A in fixtures.items():
            n = len(A)
            rp, ci, v, _ = core.dense_to_csr(A)
            ref, ref_pivot = core.ilu0_reference(rp, ci, v, n)
            gpu, gpu_pivot = ops.ilu0_factor(rp, ci, v, n, device=device, dtype=dtype)
            ok, detail = factor_parity(gpu, ref, dtype)
            check(
                f"ilu0-parity-{dtype}-{name}",
                ok and gpu_pivot == ref_pivot,
                pivot_gpu=gpu_pivot,
                pivot_ref=ref_pivot,
                **detail,
            )
            worst[dtype] = max(worst[dtype], detail["rel_l2"])
            # Dense-pattern ILU(0) must reconstruct A (exact LU, no fill dropped).
            if name.startswith("dense"):
                L, U = core.split_lu(rp, ci, gpu, n)
                recon = core.matmul_dense(L, U)
                err = max(
                    abs(recon[i][j] - A[i][j]) for i in range(n) for j in range(n)
                )
                check(
                    f"ilu0-reconstruct-{dtype}-{name}",
                    err < (1e-3 if dtype == torch.float32 else 1e-8),
                    max_abs=err,
                )
    REPORT["measurements"]["ilu0_worst_rel_l2"] = {str(k): v for k, v in worst.items()}


def ic0_parity(device):
    fixtures = {}
    for n in (32, 64):
        A = tridiagonal(n, diag=4.0, off=1.0)  # SPD
        fixtures[f"tridiag{n}"] = [
            [A[i][j] if j <= i else 0.0 for j in range(n)] for i in range(n)
        ]
    worst = {torch.float32: 0.0, torch.float64: 0.0}
    for dtype in (torch.float32, torch.float64):
        for name, lower in fixtures.items():
            n = len(lower)
            rp, ci, v, _ = core.dense_to_csr(lower)
            ref, ref_pivot = core.ic0_reference(rp, ci, v, n)
            gpu, gpu_pivot = ops.ic0_factor(rp, ci, v, n, device=device, dtype=dtype)
            ok, detail = factor_parity(gpu, ref, dtype)
            check(
                f"ic0-parity-{dtype}-{name}",
                ok and gpu_pivot == ref_pivot,
                pivot_gpu=gpu_pivot,
                pivot_ref=ref_pivot,
                **detail,
            )
            worst[dtype] = max(worst[dtype], detail["rel_l2"])
    REPORT["measurements"]["ic0_worst_rel_l2"] = {str(k): v for k, v in worst.items()}


def pivot_contract(device):
    # Structural zero diagonal and a numeric zero pivot: both must be reported.
    cases = {
        "zero-row0": [[0.0, 1.0], [1.0, 1.0]],
        "numeric-singular": [[1.0, 1.0], [1.0, 1.0]],
    }
    for name, A in cases.items():
        n = len(A)
        rp, ci, v, _ = core.dense_to_csr(A)
        _, ref_pivot = core.ilu0_reference(rp, ci, v, n)
        _, gpu_pivot = ops.ilu0_factor(rp, ci, v, n, device=device, dtype=torch.float64)
        check(
            f"ilu0-zero-pivot-{name}",
            gpu_pivot >= 0 and gpu_pivot == ref_pivot,
            pivot_gpu=gpu_pivot,
            pivot_ref=ref_pivot,
        )


def refusals(device):
    def refuses(name, fn):
        try:
            fn()
        except (ValueError, RuntimeError, TypeError):
            check(name, True)
            return
        check(name, False)

    rp, ci, v, n = core.dense_to_csr(tridiagonal(8))
    refuses("refuse-empty", lambda: ops.ilu0_factor([0], [], [], 0, device=device))
    refuses(
        "refuse-nonsquare-metadata",
        lambda: ops.ilu0_factor([0, 1, 2], [0, 1], [1.0, 1.0], 3, device=device),
    )
    refuses(
        "refuse-nonfinite",
        lambda: ops.ilu0_factor(rp, ci, [float("nan")] + v[1:], n, device=device),
    )
    refuses(
        "refuse-bad-dtype",
        lambda: ops.ilu0_factor(rp, ci, v, n, device=device, dtype=torch.int8),
    )
    refuses(
        "refuse-cpu-device",
        lambda: ops.ilu0_factor(rp, ci, v, n, device=torch.device("cpu")),
    )
    refuses(
        "refuse-col-out-of-range",
        lambda: ops.ilu0_factor([0, 2], [0, 99], [1.0, 1.0], 2, device=device),
    )
    refuses(
        "refuse-singular-precondition",
        lambda: ops.ilu_preconditioned_solve(
            [0, 2, 4],
            [0, 1, 0, 1],
            [1.0, 1.0, 1.0, 1.0],
            2,
            [1.0, 1.0],
            iters=5,
            device=device,
        ),
    )


def solve_and_gradient(device):
    rng = random.Random(99)
    # Native csrsv2 triangular-solve parity on the ILU factors (FP64).
    A = tridiagonal(64)
    n = 64
    rp, ci, v, _ = core.dense_to_csr(A)
    fac, _ = ops.ilu0_factor(rp, ci, v, n, device=device, dtype=torch.float64)
    (Lrp, Lci, Lval), (Urp, Uci, Uval) = ops.split_lu_csr(rp, ci, fac, n)
    r = [rng.uniform(-1, 1) for _ in range(n)]
    L = ops.TriangularSolver(
        Lrp,
        Lci,
        Lval,
        n,
        fill_mode=ops.FILL_MODE_LOWER,
        diag_type=ops.DIAG_TYPE_UNIT,
        device=device,
        dtype=torch.float64,
    )
    U = ops.TriangularSolver(
        Urp,
        Uci,
        Uval,
        n,
        fill_mode=ops.FILL_MODE_UPPER,
        diag_type=ops.DIAG_TYPE_NON_UNIT,
        device=device,
        dtype=torch.float64,
    )
    try:
        z_gpu = L.solve(r).cpu().tolist()
        z_ref = core.forward_sub_unit_lower(Lrp, Lci, Lval, n, r)
        ok, det = factor_parity(z_gpu, z_ref, torch.float64)
        check("csrsv2-lower-parity", ok, **det)
        w_gpu = U.solve(z_gpu).cpu().tolist()
        w_ref = core.back_sub_upper(Urp, Uci, Uval, n, z_ref)
        ok, det = factor_parity(w_gpu, w_ref, torch.float64)
        check("csrsv2-upper-parity", ok, **det)
    finally:
        L.close()
        U.close()

    # Preconditioned solve + RHS gradient (adjoint) vs the dense oracle.
    bounds = {torch.float64: (1e-6, 1e-5), torch.float32: (1e-3, 1e-3)}
    fixtures = {"tridiag64": tridiagonal(64), "general64": general_sparse(64, 7)}
    worst_resid = {}
    for dtype, (resid_bound, sol_bound) in bounds.items():
        worst_resid[str(dtype)] = 0.0
        for name, A in fixtures.items():
            rp, ci, v, n = core.dense_to_csr(A)
            b = [rng.uniform(-1, 1) for _ in range(n)]
            x, resid, _ = ops.ilu_preconditioned_solve(
                rp, ci, v, n, b, iters=200, device=device, dtype=dtype
            )
            x_ref = core.dense_solve(A, b)
            check(
                f"pcg-residual-{dtype}-{name}",
                resid < resid_bound,
                residual=resid,
                bound=resid_bound,
            )
            ok, det = factor_parity(x, x_ref, dtype)
            check(
                f"pcg-solution-{dtype}-{name}", ok and det["rel_l2"] < sol_bound, **det
            )
            worst_resid[str(dtype)] = max(worst_resid[str(dtype)], resid)
            # dL/db for L = 0.5||x||^2 is A^{-T} x; adjoint preconditioned solve.
            g, gresid, _ = ops.ilu_adjoint_solve(
                rp, ci, v, n, x_ref, iters=200, device=device, dtype=dtype
            )
            t_rp, t_ci, t_val = core.transpose_csr(rp, ci, v, n)
            g_ref = core.dense_solve(core.csr_to_dense(t_rp, t_ci, t_val, n), x_ref)
            ok, det = factor_parity(g, g_ref, dtype)
            check(
                f"rhs-gradient-{dtype}-{name}",
                ok and det["rel_l2"] < sol_bound,
                adjoint_residual=gresid,
                **det,
            )
    REPORT["measurements"]["solve_worst_residual"] = worst_resid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    environment(device)
    ilu0_parity(device)
    ic0_parity(device)
    pivot_contract(device)
    refusals(device)
    solve_and_gradient(device)
    passed = sum(c["passed"] for c in REPORT["checks"])
    total = len(REPORT["checks"])
    REPORT["summary"] = {
        "passed": passed,
        "total": total,
        "all_passed": passed == total,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(REPORT, indent=2) + "\n")
    print(
        f"SPARSE_SUMMARY passed={passed}/{total} all_passed={passed == total}",
        flush=True,
    )
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
