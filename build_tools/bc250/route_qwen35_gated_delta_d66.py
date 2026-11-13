# SPDX-License-Identifier: MIT
"""D66 opt-in native gated-delta routing for the frozen Transformers 5.2.0 hybrid.

The frozen Qwen3.5 torch fallback builds its unit-lower-triangular inverse with a
63-iteration Python loop (hundreds of sequential small kernel launches per call).
The qualified native ``torch.nn.bc250_gated_delta.chunk_gated_delta_rule`` (A06
preparation + A07 low-precision parity) replaces that loop with batched ATen
triangular solves. This helper routes the model's chunk call to the native op
explicitly; it installs no default and changes no package or release identity.

Boundaries (pre-declared, D66):
- eager first-order training only; no compile/graph, no higher-order gradients;
- the recurrent/decode (cache) path and the trainable-weight causal conv are NOT
  routed — the native frozen conv requires frozen weights, and the benchmark
  trains them;
- unsupported dtype/geometry/state inputs fall back to the original frozen
  implementation and are counted, never silently coerced;
- native output is not bitwise equal to the fallback (different solve order and
  FP32 preparation rounding); parity bounds are qualified by the D66 fixture.

Pure gate logic is importable without torch for workstation unit tests.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

FIXTURE = "route_qwen35_gated_delta_d66"
QUALIFIED_TRANSFORMERS = "5.2.0"
MODEL_TYPE = "qwen3_5_text"
KILL_SWITCH = "BC250_NATIVE_GATED_DELTA"
MAX_HEAD_WIDTH = 128
MAX_PADDED_TOKENS = 1048576
CHUNK_SIZE = 64
FLOAT_DTYPES = ("float32", "float16", "bfloat16")


# ---------------------------------------------------------------------------
# Pure logic (no torch import; unit-testable on the workstation).
# ---------------------------------------------------------------------------
def kill_switch_enabled(env: dict) -> bool:
    value = env.get(KILL_SWITCH, "1")
    if value not in ("0", "1"):
        raise ValueError(f"{KILL_SWITCH} must be 0 or 1")
    return value == "0"


def version_gate(transformers_version: str) -> bool:
    return transformers_version == QUALIFIED_TRANSFORMERS


def model_type_gate(model_type: str | None) -> bool:
    return model_type == MODEL_TYPE


def geometry_supported(
    batch: int,
    seq: int,
    heads: int,
    key_width: int,
    value_width: int,
    chunk_size: int = CHUNK_SIZE,
) -> bool:
    if chunk_size != CHUNK_SIZE:
        return False
    if min(batch, seq, heads, key_width, value_width) < 1:
        return False
    if max(key_width, value_width) > MAX_HEAD_WIDTH:
        return False
    padded = batch * heads * ((seq + CHUNK_SIZE - 1) // CHUNK_SIZE) * CHUNK_SIZE
    return padded <= MAX_PADDED_TOKENS


def dtype_supported(query_dtype: str, key_dtype: str, value_dtype: str) -> bool:
    return (
        query_dtype in FLOAT_DTYPES
        and key_dtype == query_dtype
        and value_dtype == query_dtype
    )


def routing_decision(
    *,
    query_dtype: str,
    key_dtype: str,
    value_dtype: str,
    batch: int,
    seq: int,
    heads: int,
    key_width: int,
    value_width: int,
    chunk_size: int,
    initial_state_is_none: bool,
    compiling: bool,
) -> str:
    """Return 'native' or the fallback reason; never raises on valid shapes."""
    if compiling:
        return "compiled-context"
    if not dtype_supported(query_dtype, key_dtype, value_dtype):
        return "dtype"
    if not geometry_supported(batch, seq, heads, key_width, value_width, chunk_size):
        return "geometry"
    if not initial_state_is_none:
        # The frozen fallback carries recurrent state in the decode path; the
        # native initial-state contract is FP32-only and stays out of D66 scope.
        return "initial-state"
    return "native"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Torch-dependent install/uninstall (lazy imports; runs on the board).
# ---------------------------------------------------------------------------
def _gated_delta_modules(model):
    return [
        module
        for module in model.modules()
        if type(module).__name__ == "Qwen3_5GatedDeltaNet"
    ]


def _native_call(query, key, value, g, beta, **kwargs):
    from torch.nn.bc250_gated_delta import chunk_gated_delta_rule

    return chunk_gated_delta_rule(query, key, value, g, beta, **kwargs)


def install(model, env: dict | None = None) -> dict:
    """Route every GatedDeltaNet chunk call on this model to the native op.

    Refuses (raises ValueError) unless the frozen qualified environment, the
    qwen3_5_text architecture and an actual gfx1013 CUDA device are present.
    Returns a record with the patched-layer count, fallback counters and the
    helper digest; keep it to uninstall.
    """
    import torch
    import transformers

    env = dict(os.environ if env is None else env)
    if kill_switch_enabled(env):
        raise ValueError(f"{KILL_SWITCH}=0 refuses native gated-delta routing")
    if not version_gate(transformers.__version__):
        raise ValueError(
            "Native gated-delta routing is qualified only with Transformers "
            f"{QUALIFIED_TRANSFORMERS}, found {transformers.__version__}"
        )
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if not model_type_gate(getattr(base.config, "model_type", None)):
        raise ValueError("Native gated-delta routing requires the qwen3_5_text model")
    parameter = next(base.parameters())
    if not parameter.is_cuda:
        raise ValueError("Native gated-delta routing requires a CUDA model")
    if (
        torch.cuda.get_device_properties(parameter.device).gcnArchName.split(":")[0]
        != "gfx1013"
    ):
        raise ValueError("Native gated-delta routing requires actual gfx1013")
    modules = _gated_delta_modules(base)
    if not modules:
        raise ValueError("No Qwen3_5GatedDeltaNet layers found on this model")
    record = {
        "fixture": FIXTURE,
        "helper_sha256": sha256_file(Path(__file__)),
        "transformers": transformers.__version__,
        "layers": len(modules),
        "fallbacks": {},
        "native_calls": 0,
        "originals": [],
    }
    for module in modules:
        original = module.chunk_gated_delta_rule

        def routed(
            query,
            key,
            value,
            g,
            beta,
            chunk_size=CHUNK_SIZE,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
            _original=original,
            _record=record,
        ):
            decision = routing_decision(
                query_dtype=str(query.dtype).removeprefix("torch."),
                key_dtype=str(key.dtype).removeprefix("torch."),
                value_dtype=str(value.dtype).removeprefix("torch."),
                batch=query.shape[0],
                seq=query.shape[1],
                heads=query.shape[2],
                key_width=query.shape[3],
                value_width=value.shape[3],
                chunk_size=chunk_size,
                initial_state_is_none=initial_state is None,
                compiling=torch.compiler.is_compiling(),
            )
            if decision != "native":
                _record["fallbacks"][decision] = (
                    _record["fallbacks"].get(decision, 0) + 1
                )
                return _original(
                    query,
                    key,
                    value,
                    g,
                    beta,
                    chunk_size=chunk_size,
                    initial_state=initial_state,
                    output_final_state=output_final_state,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                )
            _record["native_calls"] += 1
            return _native_call(
                query,
                key,
                value,
                g,
                beta,
                chunk_size=chunk_size,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )

        module.chunk_gated_delta_rule = routed
        record["originals"].append((module, original))
    print(
        f"[bc250] D66 native gated-delta routing {record['helper_sha256']}: "
        f"{record['layers']} layer(s), eager first-order, fallbacks counted",
        flush=True,
    )
    return record


def uninstall(record: dict) -> None:
    for module, original in record["originals"]:
        module.chunk_gated_delta_rule = original
    record["originals"] = []
