#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PLAN2-02 native NF4/QLoRA GPU qualification (runs inside the guardian).

Native gfx1013 only. Qualifies the NF4 weight-only recipe against independent
CPU references within the pre-declared bounds (tmp/nf4-plan2-02/BOUNDS.md), the
QLoRA composite step, the memory reduction, module state round-trip and the
refusal/kill-switch surface. Writes a JSON report; exits non-zero on any miss.
No native operator, dispatch, default or source change.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from nf4_plan2_02 import core, ops, reference  # noqa: E402

# Pre-declared compute parity bounds (GPU result vs CPU-double reference operand).
COMPUTE_BOUNDS = {
    torch.float16: (0.015, 0.05, 0.004),
    torch.bfloat16: (0.03, 0.08, 0.01),
    torch.float32: (1e-4, 1e-4, 1e-4),
}
CODE_MATCH_MIN = 0.999
SCALE_REL_MAX = 1e-5
DEQUANT_REL_L2_MAX = 0.01
RECIPE_REL_FROB_MAX = 0.10
PERSISTENT_RATIO_MAX = 0.30

REPORT = {"checks": [], "measurements": {}, "environment": {}}


def check(name, passed, **detail):
    REPORT["checks"].append({"name": name, "passed": bool(passed), **detail})
    if not passed:
        print(f"NF4_FAIL {name} {detail}", flush=True)
    return passed


def rel_l2(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def parity(a, b, dtype):
    rtol, atol, l2 = COMPUTE_BOUNDS[dtype]
    a = a.detach().double().cpu()
    b = b.detach().double().cpu()
    close = bool(torch.allclose(a, b, rtol=rtol, atol=atol))
    rl2 = rel_l2(a, b)
    return close and rl2 < l2, {
        "allclose": close,
        "rel_l2": rl2,
        "max_abs": (a - b).abs().max().item(),
        "rtol": rtol,
        "atol": atol,
        "l2_bound": l2,
    }


def to_list(t):
    return t.detach().double().cpu().tolist()


def environment(device):
    env = REPORT["environment"]
    env["torch"] = torch.__version__
    env["hip"] = torch.version.hip
    env["gcnArchName"] = torch.cuda.get_device_properties(device).gcnArchName
    env["device_name"] = torch.cuda.get_device_name(device)
    check(
        "gfx1013-native",
        env["gcnArchName"].startswith("gfx1013:"),
        arch=env["gcnArchName"],
    )
    check(
        "bitsandbytes-absent",
        importlib.util.find_spec("bitsandbytes") is None,
    )
    check("torchao-absent", importlib.util.find_spec("torchao") is None)


def native_boundaries(device):
    # Native packed INT4 conversion must still refuse on gfx1013 (CDNA2/MFMA).
    refused_int4 = False
    try:
        w = torch.randn(64, 64, device=device, dtype=torch.bfloat16)
        torch._convert_weight_to_int4pack(w, 64)
    except (RuntimeError, AssertionError):
        refused_int4 = True
    check("native-int4-conversion-refuses", refused_int4)
    # Native weight-int8 backward must remain unimplemented (helper supplies its own).
    refused_bwd = False
    try:
        x = torch.randn(4, 64, device=device, dtype=torch.float32, requires_grad=True)
        w = torch.randint(-127, 127, (32, 64), device=device, dtype=torch.int8)
        s = torch.ones(32, device=device, dtype=torch.float32)
        torch._weight_int8pack_mm(x, w, s).sum().backward()
    except RuntimeError:
        refused_bwd = True
    check("native-int8-backward-refuses", refused_bwd)


def pack_dequant_parity(device):
    rng = torch.Generator(device="cpu").manual_seed(20261114)
    geometries = [(16, 16, 16), (32, 64, 16), (64, 128, 64), (128, 256, 64)]
    worst_code_match = 1.0
    worst_scale_rel = 0.0
    worst_dequant_l2 = 0.0
    worst_recipe_frob = 0.0
    structural_ok = True
    for n, k, bs in geometries:
        weight = torch.randn(n, k, generator=rng, dtype=torch.float32).to(device)
        packed = ops.pack(weight, bs)
        # Independent scalar reference pack (FP64) on the same weight. ref_codes
        # is packed bytes; unpack both sides to 4-bit indices before comparing.
        ref_codes, ref_scales = reference.pack_reference(to_list(weight), bs)
        gpu_indices = core.unpack_nibbles(packed.codes.flatten().tolist(), n * k)
        ref_indices = core.unpack_nibbles(ref_codes, n * k)
        match = sum(a == b for a, b in zip(gpu_indices, ref_indices)) / (n * k)
        worst_code_match = min(worst_code_match, match)
        gpu_scales = packed.scales.flatten().tolist()
        scale_rel = max(
            abs(a - b) / max(abs(b), 1e-12) for a, b in zip(gpu_scales, ref_scales)
        )
        worst_scale_rel = max(worst_scale_rel, scale_rel)
        # Dequant parity: GPU FP32 dequant vs independent double reference.
        gpu_deq = ops.dequantize(packed, torch.float32).double().cpu()
        ref_deq = torch.tensor(
            reference.dequantize_reference(ref_codes, ref_scales, n, k, bs),
            dtype=torch.float64,
        )
        worst_dequant_l2 = max(worst_dequant_l2, rel_l2(gpu_deq, ref_deq))
        # Structural invariant: every GPU dequant value == codebook[code]*scale,
        # reproduced in the SAME FP32 arithmetic the GPU uses (FP32 codebook and
        # FP32 block scales), so the comparison is bit-exact.
        cb32 = torch.tensor(core.NF4_CODEBOOK, dtype=torch.float32)
        idx = torch.tensor(gpu_indices, dtype=torch.long).reshape(n, k)
        sc32 = packed.scales.cpu().repeat_interleave(bs, dim=1)
        structural_ok &= bool(torch.equal(gpu_deq, (cb32[idx] * sc32).double()))
        # Recipe quality vs the original FP32 weight (not a model-quality claim).
        worst_recipe_frob = max(
            worst_recipe_frob, rel_l2(gpu_deq, weight.double().cpu())
        )
    check(
        "pack-code-match", worst_code_match >= CODE_MATCH_MIN, fraction=worst_code_match
    )
    check("pack-scale-parity", worst_scale_rel < SCALE_REL_MAX, max_rel=worst_scale_rel)
    check(
        "dequant-parity", worst_dequant_l2 < DEQUANT_REL_L2_MAX, rel_l2=worst_dequant_l2
    )
    check("dequant-structural-invariant", structural_ok)
    check(
        "recipe-quantization-error",
        worst_recipe_frob < RECIPE_REL_FROB_MAX,
        rel_frobenius=worst_recipe_frob,
    )
    REPORT["measurements"]["worst_recipe_rel_frobenius"] = worst_recipe_frob


def linear_parity(device):
    rng = torch.Generator(device="cpu").manual_seed(424242)
    geometries = [
        (16, 16, 16, 4),
        (32, 64, 16, 8),
        (64, 128, 64, 16),
        (128, 256, 64, 32),
    ]
    all_ok = True
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        for n, k, bs, m in geometries:
            weight = torch.randn(n, k, generator=rng, dtype=torch.float32)
            packed = ops.pack(weight.to(device), bs)
            # Reference operand: the GPU-dequantized weight cast to double (CPU).
            w_ref = ops.dequantize(packed, torch.float32).double().cpu()
            x = torch.randn(m, k, generator=rng, dtype=torch.float32)
            bias = torch.randn(n, generator=rng, dtype=torch.float32)
            xd = x.to(device=device, dtype=dtype).requires_grad_(True)
            bd = bias.to(device=device, dtype=dtype).requires_grad_(True)
            y = ops.linear(xd, packed, bd)
            y_ref = x.double() @ w_ref.t() + bias.double()
            ok, detail = parity(y, y_ref, dtype)
            all_ok &= check(f"forward-parity-{dtype}-{n}x{k}", ok, **detail)
            # Input + bias first derivatives vs CPU-double.
            dy = torch.randn(m, n, generator=rng, dtype=torch.float32)
            y.backward(dy.to(device=device, dtype=dtype))
            dx_ref = dy.double() @ w_ref
            db_ref = dy.double().sum(dim=0)
            ok_x, det_x = parity(xd.grad, dx_ref, dtype)
            ok_b, det_b = parity(bd.grad, db_ref, dtype)
            all_ok &= check(f"input-grad-parity-{dtype}-{n}x{k}", ok_x, **det_x)
            all_ok &= check(f"bias-grad-parity-{dtype}-{n}x{k}", ok_b, **det_b)
    # Strided input is normalized (the native int8 kernel had a strided bug; the
    # NF4 path uses matmul which handles views, but prove the helper normalizes).
    weight = torch.randn(32, 64, generator=rng, dtype=torch.float32).to(device)
    packed = ops.pack(weight, 16)
    base = torch.randn(8, 128, device=device, dtype=torch.float16)
    x_view = base[:, ::2].requires_grad_(True)  # strided, K=64
    y = ops.linear(x_view, packed)
    w_ref = ops.dequantize(packed, torch.float32).double().cpu()
    ok, detail = parity(y, x_view.detach().double().cpu() @ w_ref.t(), torch.float16)
    all_ok &= check("strided-input-normalized", ok, **detail)
    return all_ok


def qlora_step(device):
    torch.manual_seed(1234)
    k, n, rank, alpha = 256, 256, 16, 32
    # BF16 adapters: the FP32-range exponent avoids the FP16 AdamW exp_avg_sq
    # underflow that diverges to NaN on small gradients (QLoRA-standard choice).
    dtype = torch.bfloat16
    w_true = torch.randn(n, k, device=device, dtype=torch.float32) * 0.05
    source = torch.nn.Linear(k, n, bias=False, device=device, dtype=dtype)
    source.requires_grad_(False)  # NF4 packs an explicitly frozen weight
    base = ops.FrozenNF4Linear.from_float(source, 64)
    # Load the true weight into the frozen NF4 base.
    packed = ops.pack(w_true.to(dtype), 64)
    base.qweight = packed.codes
    base.scales = packed.scales
    model = ops.QLoRALinear(base, rank, alpha).to(device)
    base_codes = base.qweight.clone()
    base_scales = base.scales.clone()
    a0 = model.lora_a.detach().clone()
    opt = torch.optim.AdamW([model.lora_a, model.lora_b], lr=5e-3)
    x = torch.randn(32, k, device=device, dtype=dtype)
    target = (x.float() @ w_true.t()).to(dtype)
    losses = []
    finite = True
    for _ in range(16):
        opt.zero_grad(set_to_none=True)
        y = model(x)
        loss = torch.nn.functional.mse_loss(y.float(), target.float())
        loss.backward()
        finite &= bool(torch.isfinite(model.lora_a.grad).all())
        finite &= bool(torch.isfinite(model.lora_b.grad).all())
        opt.step()
        losses.append(loss.item())
    adapter_changed = not torch.equal(a0, model.lora_a.detach())
    base_unchanged = torch.equal(base_codes, base.qweight) and torch.equal(
        base_scales, base.scales
    )
    check("qlora-finite-gradients", finite)
    check(
        "qlora-loss-decrease", losses[-1] < losses[0], first=losses[0], last=losses[-1]
    )
    check("qlora-adapter-changed", adapter_changed)
    check("qlora-base-frozen", base_unchanged)
    REPORT["measurements"]["qlora_losses"] = losses
    # Higher derivatives refuse (once_differentiable).
    refused = False
    try:
        xg = torch.randn(4, k, device=device, dtype=torch.float16, requires_grad=True)
        y = ops.linear(xg, packed)
        g = torch.autograd.grad(y.sum(), xg, create_graph=True)[0]
        torch.autograd.grad(g.sum(), xg)
    except RuntimeError:
        refused = True
    check("higher-derivative-refuses", refused)


def memory_reduction(device):
    torch.manual_seed(7)
    n = k = 1024
    weight = torch.randn(n, k, device=device, dtype=torch.float16)
    packed = ops.pack(weight, 64)
    nf4_bytes = packed.codes.numel() * 1 + packed.scales.numel() * 4
    fp16_bytes = weight.numel() * 2
    ratio = nf4_bytes / fp16_bytes
    check(
        "persistent-weight-ratio",
        ratio <= PERSISTENT_RATIO_MAX,
        ratio=ratio,
        nf4_bytes=nf4_bytes,
        fp16_bytes=fp16_bytes,
    )
    REPORT["measurements"]["persistent_weight_ratio"] = ratio

    # 24 layers so persistent weight storage dominates the per-layer dequant
    # transient (the win is capacity; a tiny model's transient can mask it).
    num_layers = 24

    def peak(model_factory):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model = model_factory()
        x = torch.randn(32, k, device=device, dtype=torch.float16, requires_grad=True)
        y = model(x)
        y.float().pow(2).mean().backward()
        peak_bytes = torch.cuda.max_memory_allocated(device)
        del model, x, y
        torch.cuda.empty_cache()
        return peak_bytes

    def nf4_model():
        layers = []
        for _ in range(num_layers):
            lin = torch.nn.Linear(k, k, bias=False, device=device, dtype=torch.float16)
            lin.requires_grad_(False)
            layers.append(ops.FrozenNF4Linear.from_float(lin, 64))
        return torch.nn.Sequential(*layers)

    def fp16_model():
        layers = []
        for _ in range(num_layers):
            lin = torch.nn.Linear(k, k, bias=False, device=device, dtype=torch.float16)
            lin.requires_grad_(False)
            layers.append(lin)
        return torch.nn.Sequential(*layers)

    nf4_peak = peak(nf4_model)
    fp16_peak = peak(fp16_model)
    peak_ratio = nf4_peak / fp16_peak
    check(
        "peak-memory-reduction",
        nf4_peak < fp16_peak,
        ratio=peak_ratio,
        nf4_peak=nf4_peak,
        fp16_peak=fp16_peak,
    )
    REPORT["measurements"]["peak_memory_ratio"] = peak_ratio
    REPORT["measurements"]["nf4_peak_bytes"] = nf4_peak
    REPORT["measurements"]["fp16_peak_bytes"] = fp16_peak


def module_state(device):
    src = torch.nn.Linear(64, 32, device=device, dtype=torch.float16)
    src.requires_grad_(False)
    mod = ops.FrozenNF4Linear.from_float(src, 16)
    x = torch.randn(4, 64, device=device, dtype=torch.float16)
    y0 = mod(x)
    state = mod.state_dict()
    restored = ops.FrozenNF4Linear.from_state_dict(state, device=device)
    y1 = restored(x)
    check("module-state-roundtrip", torch.equal(y0, y1))
    cpu_state = {
        k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in state.items()
    }
    check(
        "module-state-cpu-load",
        torch.equal(
            y0,
            ops.FrozenNF4Linear.from_state_dict(cpu_state, device="cpu").to(device)(x),
        ),
    )

    # Refusals.
    def refuses(name, fn):
        try:
            fn()
        except (ValueError, RuntimeError, TypeError):
            check(name, True)
            return
        check(name, False)

    class SubLinear(torch.nn.Linear):
        pass

    refuses(
        "refuse-subclass-source",
        lambda: ops.FrozenNF4Linear.from_float(
            SubLinear(64, 32, device=device, dtype=torch.float16)
        ),
    )
    bad = dict(state)
    bad["_extra_state"] = {**state["_extra_state"], "format_version": 99}
    refuses(
        "refuse-bad-metadata",
        lambda: ops.FrozenNF4Linear.from_state_dict(bad, device=device),
    )
    incomplete = {k: v for k, v in state.items() if k != "scales"}
    refuses(
        "refuse-incomplete-payload",
        lambda: ops.FrozenNF4Linear.from_state_dict(incomplete, device=device),
    )


def refusals(device):
    weight = torch.randn(32, 64, device=device, dtype=torch.float16)
    packed = ops.pack(weight, 16)
    x = torch.randn(4, 64, device=device, dtype=torch.float16)

    def refuses(name, fn):
        try:
            fn()
        except (ValueError, RuntimeError, TypeError):
            check(name, True)
            return
        check(name, False)

    refuses(
        "refuse-grad-weight-pack",
        lambda: ops.pack(weight.clone().requires_grad_(True), 16),
    )
    refuses(
        "refuse-nonfinite-weight",
        lambda: ops.pack(
            torch.full((32, 64), float("nan"), device=device, dtype=torch.float16), 16
        ),
    )
    refuses("refuse-bad-dtype", lambda: ops.pack(weight.to(torch.int8), 16))
    refuses(
        "refuse-block-divisibility",
        lambda: ops.pack(torch.randn(32, 65, device=device, dtype=torch.float16), 16),
    )
    refuses("refuse-bad-block-size", lambda: ops.pack(weight, 7))
    refuses(
        "refuse-oversize-weight",
        lambda: ops.pack(torch.randn(4097, 64, device=device, dtype=torch.float16), 16),
    )
    refuses(
        "refuse-nonfinite-input",
        lambda: ops.linear(
            torch.full((4, 64), float("inf"), device=device, dtype=torch.float16),
            packed,
        ),
    )
    refuses(
        "refuse-k-mismatch",
        lambda: ops.linear(
            torch.randn(4, 32, device=device, dtype=torch.float16), packed
        ),
    )
    refuses(
        "refuse-bad-bias",
        lambda: ops.linear(
            x, packed, torch.randn(31, device=device, dtype=torch.float16)
        ),
    )
    refuses(
        "refuse-work-bound",
        lambda: ops.linear(
            torch.randn(4096, 64, device=device, dtype=torch.float16),
            ops.pack(torch.randn(4096, 64, device=device, dtype=torch.float16), 16),
        ),
    )
    # Kill-switch: a non-gfx1013 CUDA device is refused (CPU storage is allowed).
    refuses(
        "refuse-cpu-linear", lambda: ops.linear(x.cpu(), ops.pack(weight.cpu(), 16))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)
    environment(device)
    native_boundaries(device)
    pack_dequant_parity(device)
    linear_parity(device)
    qlora_step(device)
    memory_reduction(device)
    module_state(device)
    refusals(device)
    passed = sum(c["passed"] for c in REPORT["checks"])
    total = len(REPORT["checks"])
    REPORT["summary"] = {
        "passed": passed,
        "total": total,
        "all_passed": passed == total,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(REPORT, indent=2) + "\n")
    print(
        f"NF4_SUMMARY passed={passed}/{total} all_passed={passed == total}", flush=True
    )
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
