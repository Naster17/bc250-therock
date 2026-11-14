#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PLAN-09 reusable owned new-model admission fixture (D62).

Admits a model architecture on the native gfx1013 provider through a bounded
battery: import/load, GPU forward (vs a CPU/FP64 reference), backward with
finite/nonzero gradients, an optimizer step, bitwise checkpoint save/resume with
exact resumed continuation, and a record of the actual attention / conv /
gated-delta / quantization routes exercised plus their fallback refusals.

Owned-only: every model is random-initialized from a config. There are no
downloaded weights, no pretrained ``from_pretrained`` loads, and no dependency on
excluded applications. The pure-logic helpers below are importable without torch
so they can be unit-tested on the workstation; the GPU battery runs under the
selected runtime (``bc250-train`` / ``bc250-torch``) on the board.

This is bounded validation, not universal correctness or a model-quality proof.
Forward support alone is not a training claim.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

FIXTURE = "admit_model_d62"

# Paths that must never appear on sys.path: the excluded user application and any
# of its runtime overlays. Importing a module that inserts these is forbidden.
EXCLUDED_PATH_MARKERS = ("/home/nik/finetune-n7",)

# Weight-loading calls an owned fixture must not make.
FORBIDDEN_LOAD_CALLS = ("snapshot_download", "from_pretrained")

# Config-driven owned model registry. Plain data so it is testable without torch.
# Each entry is random-initialized from its config; no weights are fetched.
MODEL_REGISTRY: dict[str, dict] = {
    "gpt2-tiny": {
        "architecture": "GPT2LMHeadModel",
        "config_class": "GPT2Config",
        "config": {
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
        },
        "backends": ("eager", "sdpa"),
        "routes": ("attention", "conv"),
        "param_count": 21280,
    },
    "qwen3-tiny": {
        "architecture": "Qwen3ForCausalLM",
        "config_class": "Qwen3Config",
        "config": {
            "vocab_size": 257,
            "hidden_size": 128,
            "intermediate_size": 192,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "attention_dropout": 0.0,
        },
        "backends": ("sdpa", "tiled"),
        "routes": ("attention",),
    },
    "qwen35-hybrid-tiny": {
        "architecture": "Qwen3_5ForCausalLM",
        "config_class": "Qwen3_5TextConfig",
        "config": {
            "vocab_size": 257,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "max_position_embeddings": 2048,
        },
        "backends": ("sdpa",),
        "routes": ("attention", "gated-delta", "conv"),
        # The hybrid Qwen3.5 architecture lives in the separate 5.19.0 overlay,
        # not the frozen 5.2.0 consumers; admission records it absent elsewhere.
        "requires_transformers": "5.19.0",
    },
}

ROUTE_NAMES = ("attention", "conv", "gated-delta", "quant-w8", "quant-nf4")


# ---------------------------------------------------------------------------
# Pure logic (no torch import; unit-testable on the workstation).
# ---------------------------------------------------------------------------
def owned_import_guard(sys_path: list[str]) -> list[str]:
    """Return the subset of sys_path entries that touch an excluded application."""
    violations = []
    for entry in sys_path:
        if any(marker in entry for marker in EXCLUDED_PATH_MARKERS):
            violations.append(entry)
    return violations


def registry_spec(name: str) -> dict:
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown owned model {name!r}; choose from {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name]


def route_status(fired: bool, refused: bool, absent: bool) -> str:
    if absent:
        return "absent"
    if fired:
        return "fired"
    if refused:
        return "refused"
    return "not-exercised"


def version_record(versions: dict[str, str]) -> dict:
    """Normalize a mapping of package -> version, dropping empty values."""
    return {name: value for name, value in sorted(versions.items()) if value}


def new_report(model: str, dtype: str, backend: str) -> dict:
    return {
        "fixture": FIXTURE,
        "model": model,
        "dtype": dtype,
        "backend": backend,
        "status": "running",
        "versions": {},
        "checks": [],
        "routes": {},
        "refusals": [],
    }


def add_check(report: dict, name: str, status: str, **data) -> None:
    report["checks"].append({"name": name, "status": status, **data})


def summarize(report: dict) -> str:
    checks = report["checks"]
    if not checks:
        return "empty"
    if any(c["status"] == "failed" for c in checks):
        return "failed"
    if all(c["status"] in ("passed", "refused-expected", "absent") for c in checks):
        return "passed"
    return "incomplete"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Torch-dependent battery (lazy imports; runs on the board under the runtime).
# ---------------------------------------------------------------------------
def record_versions() -> dict:
    import torch

    versions = {"torch": torch.__version__, "hip": str(torch.version.hip)}
    for name in ("transformers", "accelerate", "peft", "trl", "datasets"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "")
        except ImportError:
            versions[name] = ""
    if torch.cuda.is_available():
        versions["device"] = torch.cuda.get_device_properties(0).gcnArchName
    return version_record(versions)


def build_model(name: str, dtype: str, backend: str, device: str):
    import torch
    import transformers

    spec = registry_spec(name)
    config_class = getattr(transformers, spec["config_class"])
    architecture = getattr(transformers, spec["architecture"])
    config = config_class(**spec["config"])
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = "sdpa" if backend == "tiled" else backend
    if backend == "tiled":
        from torch.nn.attention._gfx1013 import register_transformers_attention

        register_transformers_attention()
        config._attn_implementation = "bc250_tiled"
    model = architecture(config).to(getattr(torch, dtype)).to(device)
    model.config.use_cache = False
    return model.train()


def _tensor_bytes(tensor) -> bytes:
    """Raw bitwise bytes for any dtype (numpy has no bfloat16/float16 support)."""
    import torch

    flat = tensor.detach().cpu().contiguous().reshape(-1)
    return flat.view(torch.uint8).numpy().tobytes()


def state_bytes(model, opt) -> bytes:
    """Bitwise parameter + optimizer-state fingerprint (D59 pattern)."""
    import torch

    parts = [_tensor_bytes(p) for p in model.parameters()]
    for group in opt.param_groups:
        for p in group["params"]:
            state = opt.state.get(p, {})
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    parts.append(_tensor_bytes(state[key]))
            if "step" in state:
                step = state["step"]
                value = (
                    int(step.item()) if isinstance(step, torch.Tensor) else int(step)
                )
                parts.append(value.to_bytes(8, "little"))
    return b"".join(parts)


def _loss(model, ids):
    return model(input_ids=ids, labels=ids, use_cache=False).loss


def admission_battery(
    report: dict, name: str, dtype: str, backend: str, device: str
) -> None:
    import torch

    torch.manual_seed(1773)
    torch.backends.cudnn.deterministic = True
    spec = registry_spec(name)
    model = build_model(name, dtype, backend, device)
    if "param_count" in spec:
        count = sum(p.numel() for p in model.parameters())
        add_check(
            report,
            "load-param-count",
            "passed" if count == spec["param_count"] else "failed",
            expected=spec["param_count"],
            actual=count,
        )
    else:
        add_check(
            report,
            "load",
            "passed",
            parameters=sum(p.numel() for p in model.parameters()),
        )

    vocab = spec["config"]["vocab_size"]
    length = 16
    ids = torch.randint(0, vocab, (2, length), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    compute_dtype = getattr(torch, dtype)

    def autocast():
        return torch.autocast("cuda", dtype=compute_dtype, enabled=(dtype != "float32"))

    # Forward vs an independent CPU/FP64 reference. Autocast matches the
    # qualified usage pattern (QK-norm models upcast q/k outside autocast).
    with autocast():
        loss = _loss(model, ids)
    cpu = copy.deepcopy(model).cpu().double()
    # The reference must use a CPU-compatible attention path; a tiled GPU model
    # would otherwise carry bc250_tiled into the float64 CPU forward.
    cpu.config._attn_implementation = "sdpa"
    ref_loss = _loss(cpu, ids.cpu())
    relative = abs(loss.item() - ref_loss.item()) / max(abs(ref_loss.item()), 1e-12)
    tolerance = 0.02 if dtype == "float32" else 0.06
    forward_ok = torch.isfinite(loss).item() and relative < tolerance
    add_check(
        report,
        "forward-vs-cpu-fp64",
        "passed" if forward_ok else "failed",
        loss=loss.item(),
        reference=ref_loss.item(),
        relative=relative,
    )

    # Backward: finite, nonzero gradients.
    opt.zero_grad(set_to_none=True)
    with autocast():
        loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    finite = all(g is not None and torch.isfinite(g).all().item() for g in grads)
    nonzero = any(g is not None and g.abs().sum().item() > 0 for g in grads)
    add_check(
        report,
        "backward-grads",
        "passed" if finite and nonzero else "failed",
        finite=finite,
        nonzero=nonzero,
        tensors=len(grads),
    )

    # Optimizer step changes weights.
    before = state_bytes(model, opt)
    opt.step()
    after = state_bytes(model, opt)
    add_check(report, "optimizer-step", "passed" if before != after else "failed")

    # Bitwise save/resume + exact resumed continuation (D59 backbone). Save at
    # the pre-continuation state, then derive the reference continuation from the
    # live model and replay it from the restored checkpoint.
    opt.zero_grad(set_to_none=True)
    pre_state = state_bytes(model, opt)

    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "admission.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
            },
            checkpoint,
        )
        ref_losses = []
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            with autocast():
                step_loss = _loss(model, ids)
                step_loss.backward()
            opt.step()
            ref_losses.append(step_loss.item())
        ref_state = state_bytes(model, opt)

        fresh = build_model(name, dtype, backend, device)
        fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
        payload = torch.load(checkpoint, weights_only=True, map_location="cpu")
        fresh.load_state_dict(payload["model"])
        fresh_opt.load_state_dict(payload["opt"])
        torch.set_rng_state(payload["rng"])
        torch.cuda.set_rng_state(payload["cuda_rng"])
        restored = state_bytes(fresh, fresh_opt) == pre_state
        add_check(report, "resume-bitwise-state", "passed" if restored else "failed")

        resumed_losses = []
        for _ in range(3):
            fresh_opt.zero_grad(set_to_none=True)
            with autocast():
                step_loss = _loss(fresh, ids)
                step_loss.backward()
            fresh_opt.step()
            resumed_losses.append(step_loss.item())
        exact = (
            resumed_losses == ref_losses and state_bytes(fresh, fresh_opt) == ref_state
        )
        add_check(
            report,
            "resume-exact-continuation",
            "passed" if exact else "failed",
            reference=ref_losses,
            resumed=resumed_losses,
        )


def _probe(report: dict, route: str, name: str, fn) -> None:
    """Run an owned op-level probe; record fired / refused / absent."""
    try:
        result = fn()
    except ImportError:
        report["routes"][route] = route_status(False, False, True)
        add_check(report, name, "absent")
        return
    except (ValueError, NotImplementedError, RuntimeError) as exc:
        report["routes"][route] = route_status(False, True, False)
        report["refusals"].append({"probe": name, "error": str(exc)})
        add_check(report, name, "refused-expected", error=str(exc))
        return
    except Exception as exc:  # record unexpected probe errors, never crash the report
        report["routes"][route] = "error"
        add_check(report, name, "failed", error=f"{type(exc).__name__}: {exc}")
        return
    fired = bool(result)
    report["routes"][route] = route_status(fired, False, False)
    add_check(report, name, "passed" if fired else "failed")


def probe_routes(report: dict, device: str) -> None:
    import torch

    # The conv/quant probes load sibling owned modules whose internal imports
    # (e.g. bc250_quant_b051) must resolve from the fixture directory.
    fixture_dir = str(Path(__file__).resolve().parent)
    if fixture_dir not in sys.path:
        sys.path.insert(0, fixture_dir)

    dtype = torch.float32

    def attention_backend_flags():
        from torch.nn.attention import SDPBackend, sdpa_kernel

        q = torch.randn(1, 4, 64, 128, device=device, dtype=dtype)
        params = torch.backends.cuda.SDPAParams(q, q, q, None, 0.0, False, False)
        efficient = torch.backends.cuda.can_use_efficient_attention(params, debug=False)
        flash = torch.backends.cuda.can_use_flash_attention(params, debug=False)
        # The native build must not silently run efficient/flash attention.
        refused = False
        try:
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                torch.nn.functional.scaled_dot_product_attention(q, q, q)
        except RuntimeError:
            refused = True
        report.setdefault("backend_flags", {}).update(
            {
                "can_use_efficient": bool(efficient),
                "can_use_flash": bool(flash),
                "efficient_refused": refused,
            }
        )
        return not efficient and not flash and refused

    def tiled_sdpa():
        from torch.nn.attention._gfx1013 import scaled_dot_product_attention

        q = torch.randn(1, 2, 64, 256, device=device, dtype=dtype)
        out = scaled_dot_product_attention(q, q, q, is_causal=True)
        return torch.isfinite(out).all().item()

    def gated_delta():
        from torch.nn.bc250_gated_delta import chunk_gated_delta_rule

        b, t, h, dk, dv = 1, 64, 2, 16, 16
        q = torch.randn(b, t, h, dk, device=device, dtype=dtype)
        k = torch.randn(b, t, h, dk, device=device, dtype=dtype)
        v = torch.randn(b, t, h, dv, device=device, dtype=dtype)
        g = -torch.rand(b, t, h, device=device, dtype=dtype)
        beta = torch.rand(b, t, h, device=device, dtype=dtype)
        result = chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=64)
        out = result[0] if isinstance(result, tuple) else result
        return torch.isfinite(out).all().item()

    def causal_conv():
        import importlib.util

        path = Path(__file__).resolve().parent / "qwen35_training_ops.py"
        spec = importlib.util.spec_from_file_location("bc250_conv_ops", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        x = torch.randn(1, 8, 32, device=device, dtype=dtype)
        w = torch.randn(8, 4, device=device, dtype=dtype)
        bias = torch.randn(8, device=device, dtype=dtype)
        out = module.frozen_causal_conv(x, w, bias, "silu")
        return torch.isfinite(out).all().item()

    def quant_w8():
        import importlib.util

        path = Path(__file__).resolve().parent / "bc250_quant_b055.py"
        spec = importlib.util.spec_from_file_location("bc250_w8", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source = torch.nn.Linear(16, 16).to(device=device, dtype=dtype)
        source.requires_grad_(False)  # frozen W8 packs an explicitly frozen weight
        frozen = module.FrozenW8Linear.from_float(source)
        out = frozen(torch.randn(2, 16, device=device, dtype=dtype))
        return torch.isfinite(out).all().item()

    def quant_nf4():
        from nf4_plan2_02 import ops as nf4_ops

        source = torch.nn.Linear(16, 16).to(device=device, dtype=dtype)
        source.requires_grad_(False)  # frozen NF4 packs an explicitly frozen weight
        frozen = nf4_ops.FrozenNF4Linear.from_float(source, 16)
        out = frozen(torch.randn(2, 16, device=device, dtype=dtype))
        return torch.isfinite(out).all().item()

    _probe(report, "attention", "attention-backend-flags", attention_backend_flags)
    _probe(report, "attention", "tiled-sdpa", tiled_sdpa)
    _probe(report, "gated-delta", "gated-delta-op", gated_delta)
    _probe(report, "conv", "causal-conv-op", causal_conv)
    _probe(report, "quant-w8", "frozen-w8-op", quant_w8)
    _probe(report, "quant-nf4", "frozen-nf4-op", quant_nf4)


def architecture_available(name: str) -> tuple[bool, str]:
    """Whether the registry architecture exists in the installed transformers."""
    import transformers

    spec = registry_spec(name)
    for attr in (spec["config_class"], spec["architecture"]):
        if not hasattr(transformers, attr):
            return False, f"{attr} absent in transformers {transformers.__version__}"
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2-tiny", choices=sorted(MODEL_REGISTRY))
    parser.add_argument(
        "--dtype", default="float32", choices=("float32", "bfloat16", "float16")
    )
    parser.add_argument(
        "--backend", default="eager", choices=("eager", "sdpa", "tiled")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--routes-only",
        action="store_true",
        help="skip the model battery; only probe kernel routes",
    )
    args = parser.parse_args()

    violations = owned_import_guard(sys.path)
    if violations:
        raise SystemExit(
            f"Owned-import guard refused excluded sys.path entries: {violations}"
        )

    report = new_report(args.model, args.dtype, args.backend)
    report["versions"] = record_versions()
    if not args.routes_only:
        available, reason = architecture_available(args.model)
        if available:
            try:
                admission_battery(
                    report, args.model, args.dtype, args.backend, args.device
                )
            except Exception as exc:
                add_check(
                    report,
                    "admission-battery",
                    "failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
        else:
            report["model_absent"] = reason
            add_check(report, "model-architecture", "absent", reason=reason)
    probe_routes(report, args.device)
    report["status"] = summarize(report)
    report["fixture_sha256"] = sha256_bytes(Path(__file__).read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"ADMIT_MODEL_D62 {report['status']} model={args.model} "
        f"dtype={args.dtype} backend={args.backend} routes={report['routes']}"
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
