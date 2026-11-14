# SPDX-License-Identifier: MIT
"""Independent pure-Python CPU-double NF4 reference (no torch).

This is the parity anchor for the GPU qualification. It deliberately uses a
linear-scan nearest code (core.nearest_code) and a scalar triple-loop matmul so
it shares no algorithm with the torch ops (searchsorted + rocBLAS GEMM); only
the canonical codebook constants are common. Fixtures are small enough for the
scalar reference; larger compute parity uses torch-double on the board.
"""

from . import core


def pack_reference(weight, block_size):
    """weight: [n][k] floats -> (packed bytes flat row-major, scales flat).

    Per-block absmax scale; normalized value mapped to the nearest NF4 level.
    """
    core.validate_block_size(block_size)
    n = len(weight)
    k = len(weight[0]) if n else 0
    core.validate_weight_shape((n, k))
    if k % block_size != 0:
        raise ValueError("k must be divisible by the block size")
    packed = []
    scales = []
    for row in weight:
        indices = []
        for start in range(0, k, block_size):
            block = row[start : start + block_size]
            maximum = max((abs(v) for v in block), default=0.0)
            scale = 1.0 if maximum == 0.0 else maximum
            scales.append(scale)
            for value in block:
                indices.append(core.nearest_code(value / scale))
        packed.extend(core.pack_nibbles(indices))
    return packed, scales


def dequantize_reference(packed, scales, n, k, block_size):
    """Inverse of pack_reference -> [n][k] double weights."""
    core.validate_block_size(block_size)
    if k % block_size != 0:
        raise ValueError("k must be divisible by the block size")
    weight = []
    scale_index = 0
    for _ in range(n):
        indices = core.unpack_nibbles(packed[: (k + 1) // 2], k)
        packed = packed[(k + 1) // 2 :]
        row = []
        for start in range(0, k, block_size):
            scale = scales[scale_index]
            scale_index += 1
            for j in range(start, start + block_size):
                row.append(core.NF4_CODEBOOK[indices[j]] * scale)
        weight.append(row)
    return weight


def matmul(a, b):
    """Scalar double matmul a[m][p] @ b[p][n]."""
    m = len(a)
    p = len(b)
    n = len(b[0]) if p else 0
    out = [[0.0] * n for _ in range(m)]
    for i in range(m):
        ai = a[i]
        oi = out[i]
        for t in range(p):
            a_it = ai[t]
            bt = b[t]
            for j in range(n):
                oi[j] += a_it * bt[j]
    return out


def transpose(w):
    return [list(col) for col in zip(*w)]


def nf4_forward(x, packed, scales, n, k, block_size, bias=None):
    """y = x @ W_dequant^T + bias, all in double."""
    weight = dequantize_reference(packed, scales, n, k, block_size)
    y = matmul(x, transpose(weight))
    if bias is not None:
        for row in y:
            for j in range(len(row)):
                row[j] += bias[j]
    return y


def nf4_input_grad(dy, packed, scales, n, k, block_size):
    """dx = dy @ W_dequant (base weight frozen; first input derivative)."""
    weight = dequantize_reference(packed, scales, n, k, block_size)
    return matmul(dy, weight)
