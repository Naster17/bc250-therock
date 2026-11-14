# SPDX-License-Identifier: MIT
"""Pure fixture/bounds data for PLAN2-04 (no torch import; unit-testable).

Bounds are transcribed from tmp/attention-plan2-04/BOUNDS.md, fixed before any
GPU run. Geometries are (name, B, H, HK, M, N, D, mode) with mode in
{none, causal, bool, additive}; bench rows are (name, H, HK, D, L, causal, dtype).
"""

SEED = 20261114
DTYPES = ("float32", "float16", "bfloat16")

# rel-L2 vs the independent CPU-FP64 reference, output and each Q/K/V gradient.
REL_L2_BOUNDS = {"float32": 3e-5, "float16": 5e-3, "bfloat16": 3e-2}
# assert_close vs GPU math SDPA: (rtol, atol).
CLOSE_BOUNDS = {
    "float32": (5e-4, 3e-5),
    "float16": (8e-3, 3e-3),
    "bfloat16": (5e-2, 3e-2),
}

PARITY_PART1 = [
    ("small-full-64", 1, 4, 4, 256, 256, 64, "none"),
    ("causal-64", 1, 4, 4, 512, 512, 64, "causal"),
    ("gqa-causal-128", 1, 8, 2, 512, 512, 128, "causal"),
    ("bool-mask-64", 1, 4, 4, 512, 512, 64, "bool"),
    ("additive-mask-64", 1, 4, 4, 512, 512, 64, "additive"),
    ("cross-bool-64", 1, 4, 4, 256, 512, 64, "bool"),
    ("head256-full", 1, 2, 2, 512, 512, 256, "none"),
    ("head256-causal", 1, 2, 2, 256, 256, 256, "causal"),
]
PARITY_PART2 = [
    ("causal-128-1024", 1, 8, 8, 1024, 1024, 128, "causal"),
    ("gqa-causal-128-2048", 1, 16, 4, 2048, 2048, 128, "causal"),
    ("full-64-4096", 1, 8, 8, 4096, 4096, 64, "none"),
    ("gqa-causal-128-4096", 1, 4, 2, 4096, 4096, 128, "causal"),
    ("head256-full-2048", 1, 2, 2, 2048, 2048, 256, "none"),
]

REFUSALS = [
    # (name, expected exception class name)
    ("dropout", "NotImplementedError"),
    ("mask-gradient", "NotImplementedError"),
    ("causal-plus-mask", "ValueError"),
    ("gqa-without-flag", "ValueError"),
    ("nonfinite-scale", "ValueError"),
    ("head-dim-257", "ValueError"),
    ("seq-4097", "ValueError"),
    ("dtype-float64", "ValueError"),
    ("dtype-mismatch", "ValueError"),
    ("cpu-device", "ValueError"),
    ("wrong-rank", "ValueError"),
    ("higher-order-grad", "RuntimeError"),
    ("compile-fullgraph", "Exception"),
]

WARMUPS = 2
SAMPLES = 5
DRIFT_LIMIT = 1.25


def bench_part1():
    rows = []
    for d, h, hk in ((64, 16, 16), (128, 16, 4)):
        for length in (512, 1024, 2048, 4096):
            for causal in (False, True):
                for dtype in ("float16", "bfloat16"):
                    name = f"d{d}-h{h}x{hk}-l{length}-{'c' if causal else 'f'}-{dtype}"
                    rows.append((name, h, hk, d, length, causal, dtype))
    return rows


def bench_part2():
    rows = []
    for length in (512, 1024, 2048):
        for dtype in ("float16", "bfloat16"):
            rows.append(
                (f"d256-h8x2-l{length}-c-{dtype}", 8, 2, 256, length, True, dtype)
            )
    for length in (1024, 4096):
        rows.append(
            (f"d128-h16x4-l{length}-c-float32", 16, 4, 128, length, True, "float32")
        )
    rows.append(("d64-h16x16-l4096-f-float32", 16, 16, 64, 4096, False, "float32"))
    return rows


def policy_for(dtype, head_dim, grad_enabled):
    """Mirror of the installed module's forward policy, for report labeling."""
    if dtype == "bfloat16" and head_dim > 128 and grad_enabled:
        return "hybrid-math-forward-tiled-backward"
    return "pure-tiled"


def parity_geometries(part):
    if part == 1:
        return list(PARITY_PART1)
    if part == 2:
        return list(PARITY_PART2)
    raise ValueError(f"unknown parity part {part}")


def bench_rows(part):
    if part == 1:
        return bench_part1()
    if part == 2:
        return bench_part2()
    raise ValueError(f"unknown bench part {part}")
