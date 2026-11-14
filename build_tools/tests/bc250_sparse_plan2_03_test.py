# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pure (no torch) tests for the PLAN2-03 CPU ILU(0)/IC(0) reference contract.

The GPU hipSPARSE parity runs on native gfx1013; these tests prove the
dependency-free reference is correct (dense-pattern ILU(0) == exact LU, dense SPD
IC(0) == exact Cholesky), the CSR helpers, pivot detection and the dense oracle.
"""

from pathlib import Path
import sys
import unittest

BC250 = Path(__file__).resolve().parents[1] / "bc250"
if str(BC250) not in sys.path:
    sys.path.insert(0, str(BC250))

from sparse_plan2_03 import core  # noqa: E402


def max_abs_diff(A, B):
    return max(abs(A[i][j] - B[i][j]) for i in range(len(A)) for j in range(len(A)))


def hilbert_plus(n, shift):
    return [
        [1.0 / (i + j + 1) + (shift if i == j else 0.0) for j in range(n)]
        for i in range(n)
    ]


class CsrHelperTest(unittest.TestCase):
    def test_round_trip(self):
        A = [[2.0, 0.0, 1.0], [0.0, 3.0, 0.0], [1.0, 0.0, 4.0]]
        rp, ci, v, n = core.dense_to_csr(A)
        self.assertEqual(core.csr_to_dense(rp, ci, v, n), A)
        self.assertEqual(rp, [0, 2, 3, 5])

    def test_matvec(self):
        A = [[2.0, 0.0, 1.0], [0.0, 3.0, 0.0], [1.0, 0.0, 4.0]]
        rp, ci, v, n = core.dense_to_csr(A)
        x = [1.0, 2.0, 3.0]
        self.assertEqual(core.csr_matvec(rp, ci, v, n, x), [5.0, 6.0, 13.0])


class Ilu0Test(unittest.TestCase):
    def test_dense_pattern_equals_exact_lu(self):
        for n in (3, 4, 5):
            A = hilbert_plus(n, n)
            rp, ci, v, _ = core.dense_to_csr(A)
            fac, pivot = core.ilu0_reference(rp, ci, v, n)
            self.assertEqual(pivot, -1)
            L, U = core.split_lu(rp, ci, fac, n)
            self.assertLess(max_abs_diff(core.matmul_dense(L, U), A), 1e-10)

    def test_sparse_no_fill_pattern_preserved(self):
        # Tridiagonal: ILU(0) keeps the tridiagonal pattern (no fill).
        A = [[4.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 4.0]]
        rp, ci, v, n = core.dense_to_csr(A)
        fac, pivot = core.ilu0_reference(rp, ci, v, n)
        self.assertEqual(pivot, -1)
        self.assertEqual(len(fac), len(v))  # no new entries
        L, U = core.split_lu(rp, ci, fac, n)
        # L@U differs from A only by dropped fill at (0,2)/(2,0).
        LU = core.matmul_dense(L, U)
        self.assertAlmostEqual(LU[0][0], A[0][0], places=10)
        self.assertAlmostEqual(LU[1][1], A[1][1], places=10)

    def test_zero_pivot_detected(self):
        A = [[0.0, 1.0], [1.0, 1.0]]  # structural zero diagonal at row 0
        rp, ci, v, n = core.dense_to_csr(A)
        _, pivot = core.ilu0_reference(rp, ci, v, n)
        self.assertEqual(pivot, 0)

    def test_numeric_zero_pivot(self):
        # Row 1 becomes a zero diagonal after elimination: [[1,1],[1,1]].
        A = [[1.0, 1.0], [1.0, 1.0]]
        rp, ci, v, n = core.dense_to_csr(A)
        _, pivot = core.ilu0_reference(rp, ci, v, n)
        self.assertEqual(pivot, 1)


class Ic0Test(unittest.TestCase):
    def test_dense_spd_equals_exact_cholesky(self):
        for n in (3, 4):
            B = hilbert_plus(n, 0.5)
            # A = B^T B + n*I is SPD with a complete lower-triangle pattern.
            BT = [[B[j][i] for j in range(n)] for i in range(n)]
            BTB = core.matmul_dense(BT, B)
            A = [
                [BTB[i][j] + (n if i == j else 0.0) for j in range(n)] for i in range(n)
            ]
            lower = [[A[i][j] if j <= i else 0.0 for j in range(n)] for i in range(n)]
            rp, ci, v, _ = core.dense_to_csr(lower)
            fac, pivot = core.ic0_reference(rp, ci, v, n)
            self.assertEqual(pivot, -1)
            L = [[0.0] * n for _ in range(n)]
            for i in range(n):
                for p in range(rp[i], rp[i + 1]):
                    L[i][ci[p]] = fac[p]
            LT = [[L[j][i] for j in range(n)] for i in range(n)]
            self.assertLess(max_abs_diff(core.matmul_dense(L, LT), A), 1e-9)

    def test_nonpositive_pivot_detected(self):
        A = [[1.0, 2.0], [2.0, 1.0]]  # indefinite lower triangle
        lower = [[1.0, 0.0], [2.0, 1.0]]
        rp, ci, v, n = core.dense_to_csr(lower)
        _, pivot = core.ic0_reference(rp, ci, v, n)
        self.assertEqual(pivot, 1)


class DenseOracleTest(unittest.TestCase):
    def test_solve_reconstructs_rhs(self):
        A = hilbert_plus(4, 4)
        b = [1.0, 2.0, 3.0, 4.0]
        x = core.dense_solve(A, b)
        Ax = [sum(A[i][j] * x[j] for j in range(4)) for i in range(4)]
        self.assertLess(max(abs(Ax[i] - b[i]) for i in range(4)), 1e-10)

    def test_singular_raises(self):
        with self.assertRaises(ValueError):
            core.dense_solve([[1.0, 1.0], [1.0, 1.0]], [1.0, 2.0])


def _split_lu(rp, ci, fac, n):
    Lrp, Lci, Lv = [0], [], []
    Urp, Uci, Uv = [0], [], []
    for i in range(n):
        for p in range(rp[i], rp[i + 1]):
            if ci[p] < i:
                Lci.append(ci[p])
                Lv.append(fac[p])
        Lci.append(i)
        Lv.append(1.0)
        Lrp.append(len(Lv))
        for p in range(rp[i], rp[i + 1]):
            if ci[p] >= i:
                Uci.append(ci[p])
                Uv.append(fac[p])
        Urp.append(len(Uv))
    return (Lrp, Lci, Lv), (Urp, Uci, Uv)


class SolveReferenceTest(unittest.TestCase):
    def test_lu_substitution_solves_system(self):
        A = hilbert_plus(4, 4)
        rp, ci, v, n = core.dense_to_csr(A)
        fac, piv = core.ilu0_reference(rp, ci, v, n)
        self.assertEqual(piv, -1)
        (Lrp, Lci, Lv), (Urp, Uci, Uv) = _split_lu(rp, ci, fac, n)
        b = [1.0, 2.0, 3.0, 4.0]
        z = core.forward_sub_unit_lower(Lrp, Lci, Lv, n, b)
        x = core.back_sub_upper(Urp, Uci, Uv, n, z)
        Ax = core.csr_matvec(rp, ci, v, n, x)
        self.assertLess(max(abs(Ax[i] - b[i]) for i in range(n)), 1e-9)

    def test_transpose_csr_matvec(self):
        A = [[1.0, 2.0, 0.0], [0.0, 3.0, 4.0], [5.0, 0.0, 6.0]]
        rp, ci, v, n = core.dense_to_csr(A)
        trp, tci, tv = core.transpose_csr(rp, ci, v, n)
        x = [1.0, 2.0, 3.0]
        y = core.csr_matvec(trp, tci, tv, n, x)
        AT = [[A[j][i] for j in range(n)] for i in range(n)]
        expected = [sum(AT[i][j] * x[j] for j in range(n)) for i in range(n)]
        self.assertEqual(y, expected)


class OrchestratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib.util

        nf4 = BC250 / "nf4_plan2_02"
        if str(nf4) not in sys.path:
            sys.path.insert(0, str(nf4))
        spec = importlib.util.spec_from_file_location(
            "run_sparse", BC250 / "sparse_plan2_03" / "run_sparse.py"
        )
        cls.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.runner)

    def test_guardian_config(self):
        cfg = self.runner.guardian_config("sparse-r1")
        self.assertEqual(cfg["mode"], "single")
        self.assertEqual(cfg["rank"], 0)
        self.assertEqual(cfg["label"], "sparse-r1")
        self.assertEqual(cfg["audit"], self.runner.REPORT)
        self.assertIn(self.runner.QUALIFY_SCRIPT, cfg["argv"])
        self.assertIn("--output", cfg["argv"])
        self.assertTrue(cfg["evidence"].endswith("sparse-r1"))

    def test_package_files_complete(self):
        for name in self.runner.PACKAGE_FILES:
            self.assertTrue((BC250 / "sparse_plan2_03" / name).exists(), name)

    def test_reuses_qualified_guardian(self):
        self.assertTrue((self.runner.NF4_DIR / "guard_nf4.py").exists())
        self.assertTrue((self.runner.base.SOAK_DIR / "bounded_launch.py").exists())
        self.assertTrue((self.runner.base.SOAK_DIR / "kernel_noise.py").exists())


if __name__ == "__main__":
    unittest.main()
