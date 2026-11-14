# SPDX-License-Identifier: MIT
"""PLAN2-02 native NF4/QLoRA weight-only quantization for gfx1013.

Pure-Python core/reference (no torch) plus a torch ops module that dequantizes
4-bit NormalFloat weights to FP16/BF16/FP32 and computes through the qualified
rocBLAS GEMM. NF4 is a memory win, not a compute win, on RDNA1 (no INT4 matrix
cores). Explicit opt-in; no native operator, dispatch or default change.
"""
