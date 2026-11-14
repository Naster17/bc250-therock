# SPDX-License-Identifier: MIT
"""Pure NF4 constants and scalar helpers (no torch).

The codebook is the canonical 16-level NormalFloat-4 table used by bitsandbytes
NF4: the quantiles of a standard normal, normalized so the extreme levels are
exactly -1 and +1. This module is dependency-free so the quantization contract
and an independent reference are testable on the workstation; the torch ops and
the GPU parity qualification live in ops.py and run on native gfx1013.

This is an NF4-compatible recipe qualified against an independent CPU-double
reference. It is NOT a bitsandbytes bit-exactness claim (bitsandbytes is absent).
"""

import bisect

# Canonical NormalFloat-4 levels, ascending, in [-1, 1]. Index 7 is exactly 0.
NF4_CODEBOOK = (
    -1.0,
    -0.6961111111114001,
    -0.5251111111189117,
    -0.39411111111172876,
    -0.2841111111169861,
    -0.18411111111132605,
    -0.0910111111100267,
    0.0,
    0.07951111111154178,
    0.16011111111124402,
    0.24611111111149664,
    0.33711111111196185,
    0.4401111111153259,
    0.5621111111150305,
    0.7221111111126917,
    1.0,
)

# Midpoints between consecutive levels: the nearest-code decision boundaries.
NF4_MIDPOINTS = tuple((NF4_CODEBOOK[i] + NF4_CODEBOOK[i + 1]) / 2.0 for i in range(15))

BLOCK_SIZES = (16, 32, 64, 128, 256)
DEFAULT_BLOCK_SIZE = 64

# str(torch.dtype) names, matching the module extra-state metadata.
COMPUTE_DTYPES = ("torch.float16", "torch.bfloat16", "torch.float32")

# Operand/work bounds (mirror the qualified B05.x W8 prototype). These concern
# operand sizes and multiply-accumulates per call, not total allocator memory.
MAX_WEIGHT_ELEMENTS = 1 << 22
MAX_ACTIVATION_ELEMENTS = 1 << 20
MAX_MACS = 1 << 27
MAX_DIMENSION = 4096

FORMAT_NAME = "nf4_block_absmax"
FORMAT_VERSION = 1


def validate_block_size(block_size) -> None:
    if type(block_size) is not int or block_size not in BLOCK_SIZES:
        raise ValueError(f"NF4 block size must be one of {BLOCK_SIZES}")


def validate_weight_shape(shape) -> None:
    if len(shape) != 2 or not all(0 < d <= MAX_DIMENSION for d in shape):
        raise ValueError("Weight must be a nonempty 2D matrix with dimensions <=4096")
    if shape[0] * shape[1] > MAX_WEIGHT_ELEMENTS:
        raise ValueError("Weight exceeds the prototype storage bound")


def nearest_code(value: float) -> int:
    """Nearest codebook level via the sorted midpoints; ties round up.

    This is the same decision rule as the torch ops' searchsorted(right=True),
    so the scalar reference and the GPU pack agree by construction. The local
    tests cross-check it against a distance argmin away from the boundaries.
    """
    return bisect.bisect_right(NF4_MIDPOINTS, value)


def pack_nibbles(indices) -> list:
    """Pack two 4-bit codes per byte: low nibble = even index, high = odd.

    The final odd slot (when the count is odd) is zero-padded in the high nibble.
    """
    out = []
    for i in range(0, len(indices), 2):
        low = indices[i] & 0xF
        high = (indices[i + 1] & 0xF) if i + 1 < len(indices) else 0
        out.append(low | (high << 4))
    return out


def unpack_nibbles(packed, count: int) -> list:
    """Inverse of pack_nibbles, restoring exactly `count` 4-bit codes."""
    out = []
    for byte in packed:
        out.append(byte & 0xF)
        out.append((byte >> 4) & 0xF)
    return out[:count]


def block_absmax_scales(rows, block_size: int) -> list:
    """Per-block absolute-maximum scales for a flat row-major weight list.

    Zero blocks use scale 1 (mirrors the W8 zero-row rule). Returns one FP32
    scale per block in row-major order; the caller owns the float precision.
    """
    scales = []
    for start in range(0, len(rows), block_size):
        block = rows[start : start + block_size]
        maximum = max((abs(v) for v in block), default=0.0)
        scales.append(1.0 if maximum == 0.0 else maximum)
    return scales


def metadata_schema() -> dict:
    return dict(
        format_version=int,
        format=str,
        in_features=int,
        out_features=int,
        block_size=int,
        compute_dtype=str,
        bias=bool,
        bias_trainable=bool,
    )


def validate_metadata(state) -> dict:
    schema = metadata_schema()
    if (
        type(state) is not dict
        or set(state) != set(schema)
        or any(type(state[k]) is not t for k, t in schema.items())
    ):
        raise ValueError("Invalid frozen-NF4 metadata schema")
    if state["format_version"] != FORMAT_VERSION or state["format"] != FORMAT_NAME:
        raise ValueError("Unsupported frozen-NF4 format version")
    validate_weight_shape((state["out_features"], state["in_features"]))
    validate_block_size(state["block_size"])
    if state["in_features"] % state["block_size"] != 0:
        raise ValueError("in_features must be divisible by the NF4 block size")
    if state["compute_dtype"] not in COMPUTE_DTYPES or (
        not state["bias"] and state["bias_trainable"]
    ):
        raise ValueError("Invalid frozen-NF4 dtype or bias metadata")
    return state
