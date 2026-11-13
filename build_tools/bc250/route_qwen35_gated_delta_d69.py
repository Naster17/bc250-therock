# SPDX-License-Identifier: MIT
"""D69 opt-in native gated-delta routing for the app's Transformers 5.22.1.

Transformers 5.22.1 replaced the 5.2.0 63-iteration inverse loop with batched
``solve_triangular`` but kept the sequential Python chunk scan, and its
``Qwen3_5GatedDeltaNet.forward`` calls the MODULE-LEVEL global
``torch_chunk_gated_delta_rule`` (kernel-hub decorated), not a per-instance
attribute. The D66 instance patch therefore does not apply; this helper patches
the module global with a gated wrapper and restores it exactly on uninstall.

Two native targets, both explicit:
- "installed": the release's torch.nn.bc250_gated_delta (pre-D68). Its default
  path is mathematically the same ATen-solve + Python-scan structure as the
  5.22.1 reference, so little speedup is expected; it is the parity baseline.
- "fused": a staged D68 candidate module (use_fused_solve_scan), the actual
  acceleration path (fused inverse + fused scan from two chunks up).

Boundaries (pre-declared, D69; D66 discipline):
- eager first-order training only; no compile/graph, no higher-order gradients;
- the recurrent/decode (cache) path and the trainable-weight causal conv are
  NOT routed;
- unsupported dtype/geometry/state/extra-kwargs inputs fall back to the
  original 5.22.1 implementation and are counted, never silently coerced;
- the patch is module-scope: it applies to every Qwen3_5GatedDeltaNet instance
  in the process. The caller (training wrapper) verifies the app config's
  model_type is qwen3_5_text BEFORE installing;
- native output is not bitwise equal to the reference; parity bounds are
  qualified by the D69 fixture.

Pure gate logic is importable without torch for workstation unit tests.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

FIXTURE = "route_qwen35_gated_delta_d69"
QUALIFIED_TRANSFORMERS = "5.22.1"
MODEL_TYPE = "qwen3_5_text"
KILL_SWITCH = "BC250_NATIVE_GATED_DELTA"
MAX_HEAD_WIDTH = 128
MAX_PADDED_TOKENS = 1048576
CHUNK_SIZE = 64
FLOAT_DTYPES = ("float32", "float16", "bfloat16")
NATIVE_TARGETS = ("installed", "fused")
# TransformersKwargs control flags that the 5.22.1 torch chunk reference
# ignores (its **kwargs is unused in the body): output-selection flags and the
# Trainer's loss-normalization counter. Dropping them at the op level matches
# the reference exactly; the dropped keys are recorded for evidence.
IGNORED_CONTROL_KWARGS = frozenset(
    {
        "output_hidden_states",
        "output_attentions",
        "output_router_logits",
        "num_items_in_batch",
        "return_dict",
        # Cache management belongs to the model's Cache object, not the chunk
        # op; the 5.22.1 torch reference ignores this flag in the chunk body.
        "use_cache",
    }
)
# Value-sensitive kwargs: None is the ordinary unpacked/causal contract; any
# actual value signals packed/positioned/non-causal semantics the native op
# does not support -> counted fallback.
VALUE_SENSITIVE_KWARGS = (
    "cu_seq_lens_k",
    "max_length_q",
    "max_length_k",
    "position_ids",
    "seq_idx",
)


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
    extra_kwargs: bool = False,
) -> str:
    """Return 'native' or the fallback reason; never raises on valid shapes."""
    if compiling:
        return "compiled-context"
    if extra_kwargs:
        # 5.22.1 forwards cu_seqlens/**kwargs (packed sequences etc.); the
        # native op has no such contract.
        return "kwargs"
    if not dtype_supported(query_dtype, key_dtype, value_dtype):
        return "dtype"
    if not geometry_supported(batch, seq, heads, key_width, value_width, chunk_size):
        return "geometry"
    if not initial_state_is_none:
        return "initial-state"
    return "native"


def native_target_gate(target: str) -> bool:
    return target in NATIVE_TARGETS


def classify_extra_kwargs(kwargs: dict) -> tuple[str | None, list]:
    """Classify the residual kwargs of a 5.22.1 chunk call.

    Returns (fallback_reason or None, ignored_control_keys). None means the
    call fits the native unpacked/causal contract once the ignored control
    flags (which the torch reference itself ignores) are dropped.
    """
    remaining = dict(kwargs)
    if remaining.pop("cu_seqlens", None) is not None:
        return "cu-seqlens", []
    for key in VALUE_SENSITIVE_KWARGS:
        if remaining.pop(key, None) is not None:
            return "packed-kwargs", []
    if remaining.pop("is_causal", None) is False:
        return "non-causal", []
    ignored = sorted(k for k in remaining if k in IGNORED_CONTROL_KWARGS)
    for key in ignored:
        remaining.pop(key)
    if remaining:
        return "kwargs", []
    return None, ignored


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def app_model_type(config_path: Path) -> str | None:
    """Read model_type from a HF config.json (text_config nested or top)."""
    import json

    data = json.loads(Path(config_path).read_text())
    text = data.get("text_config", data)
    return text.get("model_type")


# ---------------------------------------------------------------------------
# Torch-dependent install/uninstall (lazy imports; runs on the board).
# ---------------------------------------------------------------------------
def _load_native(target: str, candidate_path: str | None):
    if target == "installed":
        import torch.nn.bc250_gated_delta as native

        return native, False
    if not candidate_path:
        raise ValueError("fused target requires the staged candidate module path")
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gated_delta_d69_candidate", candidate_path
    )
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    if not hasattr(native, "FUSED_SCAN_MIN_CHUNKS"):
        raise ValueError("candidate module lacks the D68 fused path")
    return native, True


def install(
    *,
    target: str = "fused",
    candidate_path: str | None = None,
    env: dict | None = None,
):
    """Patch the 5.22.1 modeling module global with the gated native wrapper.

    Refuses unless transformers is exactly the qualified version and the
    current CUDA device is actual gfx1013. Returns a record for uninstall.
    """
    import torch
    import transformers

    env = dict(os.environ if env is None else env)
    if kill_switch_enabled(env):
        raise ValueError(f"{KILL_SWITCH}=0: routing disabled")
    if not version_gate(transformers.__version__):
        raise ValueError(
            f"transformers {transformers.__version__} != {QUALIFIED_TRANSFORMERS}"
        )
    if not native_target_gate(target):
        raise ValueError(f"unknown native target {target!r}")
    if not torch.cuda.is_available():
        raise ValueError("CUDA device required")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch != "gfx1013":
        raise ValueError(f"device {arch} is not gfx1013")
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling

    native, fused = _load_native(target, candidate_path)
    original = modeling.torch_chunk_gated_delta_rule
    record = {
        "fixture": FIXTURE,
        "target": target,
        "fused": fused,
        "module": modeling,
        "original": original,
        "native_module": getattr(native, "__file__", "installed"),
        "native_calls": 0,
        "fallbacks": {},
        "kill_switch": KILL_SWITCH,
    }

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
        **kwargs,
    ):
        # 5.22.1 always forwards cu_seqlens (None for unpacked batches) plus
        # TransformersKwargs control flags. classify_extra_kwargs separates
        # the harmless control flags (which the torch reference ignores) from
        # packed/positioned/non-causal semantics and unknown kwargs.
        kwarg_reason, ignored = classify_extra_kwargs(kwargs)
        if ignored:
            record.setdefault("ignored_control_kwargs", set()).update(ignored)
        if kill_switch_enabled(os.environ):
            decision = "kill-switch"
        elif kwarg_reason is not None:
            decision = kwarg_reason
        else:
            decision = routing_decision(
                query_dtype=str(query.dtype).replace("torch.", ""),
                key_dtype=str(key.dtype).replace("torch.", ""),
                value_dtype=str(value.dtype).replace("torch.", ""),
                batch=query.shape[0],
                seq=query.shape[1],
                heads=query.shape[2],
                key_width=query.shape[3],
                value_width=value.shape[3],
                chunk_size=chunk_size,
                initial_state_is_none=initial_state is None,
                compiling=torch.compiler.is_compiling(),
                extra_kwargs=False,
            )
        if decision != "native":
            record["fallbacks"][decision] = record["fallbacks"].get(decision, 0) + 1
            if decision == "kwargs":
                record["fallback_kwarg_keys"] = sorted(kwargs)
            return original(
                query,
                key,
                value,
                g=g,
                beta=beta,
                chunk_size=chunk_size,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                **kwargs,
            )
        record["native_calls"] += 1
        call_kwargs = {}
        if fused:
            call_kwargs["use_fused_solve_scan"] = True
        return native.chunk_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            chunk_size=chunk_size,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            **call_kwargs,
        )

    modeling.torch_chunk_gated_delta_rule = routed
    record["wrapper"] = routed
    record["installed"] = True
    return record


def uninstall(record: dict) -> None:
    modeling = record["module"]
    if modeling.torch_chunk_gated_delta_rule is not record["wrapper"]:
        raise ValueError("module global is not the D69 wrapper; refusing")
    modeling.torch_chunk_gated_delta_rule = record["original"]
    record["installed"] = False


def install_for_app(
    app_config_path: Path,
    *,
    target: str = "fused",
    candidate_path: str | None = None,
    env: dict | None = None,
):
    """Verify the app config's model_type, then install the module route."""
    model_type = app_model_type(Path(app_config_path) / "config.json")
    if not model_type_gate(model_type):
        raise ValueError(f"app model_type {model_type!r} != {MODEL_TYPE}")
    return install(target=target, candidate_path=candidate_path, env=env)
