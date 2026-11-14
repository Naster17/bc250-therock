# SPDX-License-Identifier: MIT
"""PLAN2-03 native sparse factorization for gfx1013.

Pure-Python CPU ILU(0)/IC(0) reference (no torch) plus a ctypes hipSPARSE ops
module (csrilu02/csric02 factorization + csrsv2 triangular solves) and a bounded
ILU-preconditioned iterative solve. Native general sparse factorization is a
scalable alternative to the B02.1 dense-LU fallback; explicit opt-in, no native
operator/dispatch/default change. Nonsymmetric eig retains host-LAPACK (N2.6).
"""
