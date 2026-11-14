# SPDX-License-Identifier: MIT
"""Pure fixture/bounds data for PLAN2-05 (no torch import; unit-testable).

Bounds transcribed from tmp/compile-graph-plan2-05/BOUNDS.md, fixed before any
GPU run.
"""

SEED = 20261114
DTYPES = ("float32", "float16", "bfloat16")

# Part A: rel-L2 vs independent CPU-FP64 anchor, and compiled-vs-eager GPU.
COMPILE_CPU_BOUND = {"float32": 5e-4, "float16": 6e-3, "bfloat16": 4e-2}
COMPILE_EAGER_BOUND = {"float32": 1e-5, "float16": 2e-3, "bfloat16": 1e-2}

# Part B: multi-update graph parity (FP32 tight; low-precision autocast looser).
GRAPH_FP32_BOUND = 1e-5
GRAPH_FP16_BOUND = 2e-3
GRAPH_BF16_BOUND = 1e-2
GRAPH_UPDATES = 8
RESUME_UPDATES = 16

WARMUPS = 2
SAMPLES = 5

# Part A fixture geometries (owned, synthetic).
ATTN_SHAPE = dict(batch=2, heads=8, kv_heads=8, seq=256, head_dim=64)
ATTN_SHAPE2 = dict(batch=3, heads=8, kv_heads=8, seq=192, head_dim=64)
LORA_SHAPE = dict(batch=4, seq=32, in_features=64, out_features=64, rank=8)
EMBED_SHAPE = dict(batch=2, seq=24, vocab=128, hidden=64)
DYNAMIC_LENGTHS = (128, 192, 256)

# Part B LoRA-pattern stack geometry (launch-bound, owned).
STACK_LAYERS = 24
STACK_WIDTH = 128
STACK_RANK = 8
STACK_BATCH = 8
STACK_SEQ = 32


def graph_bound(dtype_name):
    return {
        "float32": GRAPH_FP32_BOUND,
        "float16": GRAPH_FP16_BOUND,
        "bfloat16": GRAPH_BF16_BOUND,
    }[dtype_name]


def compile_fixtures(part):
    if part == 1:
        return ["attention", "lora"]
    if part == 2:
        return ["embed_ce", "dynamic"]
    raise ValueError(f"unknown compile part {part}")
