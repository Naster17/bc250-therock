# SPDX-License-Identifier: MIT
"""Pure fixture/bounds data for PLAN2-07 (no torch import; unit-testable).

Transcribed from tmp/peft-fp16-plan2-07/BOUNDS.md, fixed before any GPU run. The
GPT2-tiny config is the D26/D28 owned identity (21280 base params); LoRA rank 4 /
alpha 8 on the 8 Conv1D projections gives 3072 trainable adapter params.
"""

SEED = 1773
UPDATES = 8
RESUME_STEPS = 3

GPT2_CONFIG = {
    "vocab_size": 97,
    "n_positions": 32,
    "n_embd": 32,
    "n_layer": 2,
    "n_head": 4,
    "n_inner": 64,
    "resid_pdrop": 0.0,
    "embd_pdrop": 0.0,
    "attn_pdrop": 0.0,
    "use_cache": False,
}
BASE_PARAM_COUNT = 21280
LORA_RANK = 4
LORA_ALPHA = 8
LORA_TARGETS = ("c_attn", "c_proj", "c_fc")
TRAINABLE_PARAM_COUNT = 3072

# Frozen-helper precision contract (FP16 working, FP32 master/frozen/reduce).
PRECISION = {
    "working_parameters": "float16",
    "master_parameters": "float32",
    "reductions": "float32",
    "buffers": "float32",
    "optimizer_state": "float32",
    "grad_scaler": True,
}

# Parity bounds vs CPU-FP64 (D25/D26/D30 gates).
LOSS_REL_BOUND = 0.02
GRAD_GLOBAL_REL_BOUND = 0.02
GRAD_PER_PARAM_REL_BOUND = 0.03
CLIP_NORM_REL_BOUND = 1e-5
MAX_GRAD_NORM = 1.0
LEARNING_RATE = 1e-3


def relative_norm(actual, reference):
    """Relative L2 norm of (actual - reference) vs reference (pure floats)."""
    num = sum((a - b) ** 2 for a, b in zip(actual, reference)) ** 0.5
    den = sum(b * b for b in reference) ** 0.5
    return num / max(den, 1e-30)
