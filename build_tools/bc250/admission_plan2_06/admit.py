#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PLAN2-06 broader model admission battery (runs on the board under guardian).

Generalizes the D62 admission battery to non-LM architectures by parameterizing
the loss/inputs per model. Imports the qualified D62 helpers (state fingerprint,
route status, report/summarize, versions) — the D62 fixture itself is untouched.
Owned pure-torch models only; no weights fetched; no diffusers.
"""

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

import admit_model_d62 as d62  # noqa: E402
from admission_plan2_06 import models  # noqa: E402

MODEL_REGISTRY = {
    "unet2d-tiny": {
        "family": "unet",
        "dtypes": ("float32", "bfloat16"),
        "routes": ("conv", "attention"),
    },
    "rnn-tiny": {
        "family": "rnn",
        "cell": "RNN",
        "dtypes": ("float32",),
        "routes": ("recurrent",),
    },
    "lstm-tiny": {
        "family": "rnn",
        "cell": "LSTM",
        "dtypes": ("float32", "bfloat16"),
        "routes": ("recurrent",),
    },
    "gru-tiny": {
        "family": "rnn",
        "cell": "GRU",
        "dtypes": ("float32",),
        "routes": ("recurrent",),
    },
}


def build_model(name, dtype, device):
    spec = MODEL_REGISTRY[name]
    if spec["family"] == "unet":
        model = models.UNet2D()
    else:
        model = models.RecurrentLM(spec["cell"])
    return model.to(getattr(torch, dtype)).to(device).train()


def make_inputs(name, device, dtype):
    if MODEL_REGISTRY[name]["family"] == "unet":
        return models.unet_inputs(device, getattr(torch, dtype))
    return models.lm_inputs(device)


def cpu_inputs(name, inputs):
    if MODEL_REGISTRY[name]["family"] == "unet":
        x, target = inputs
        return x.cpu().double(), target.cpu().double()
    return inputs.cpu()


def loss_fn(name, model, inputs):
    if MODEL_REGISTRY[name]["family"] == "unet":
        return models.unet_loss(model, inputs)
    return models.lm_loss(model, inputs)


def model_routes(name, model):
    found = set()
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            found.add("conv")
        if isinstance(module, models.SpatialAttention):
            found.add("attention")
        if isinstance(module, (torch.nn.RNN, torch.nn.LSTM, torch.nn.GRU)):
            found.add("recurrent")
    return found


def admission_battery(report, name, dtype, device):
    torch.manual_seed(1773)
    torch.backends.cudnn.deterministic = True
    spec = MODEL_REGISTRY[name]
    model = build_model(name, dtype, device)
    d62.add_check(
        report, "load", "passed", parameters=sum(p.numel() for p in model.parameters())
    )
    inputs = make_inputs(name, device, dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    compute_dtype = getattr(torch, dtype)

    def autocast():
        return torch.autocast("cuda", dtype=compute_dtype, enabled=(dtype != "float32"))

    with autocast():
        loss = loss_fn(name, model, inputs)
    cpu = copy.deepcopy(model).cpu().double()
    ref_loss = loss_fn(name, cpu, cpu_inputs(name, inputs))
    relative = abs(loss.item() - ref_loss.item()) / max(abs(ref_loss.item()), 1e-12)
    tolerance = 0.02 if dtype == "float32" else 0.06
    forward_ok = torch.isfinite(loss).item() and relative < tolerance
    d62.add_check(
        report,
        "forward-vs-cpu-fp64",
        "passed" if forward_ok else "failed",
        loss=loss.item(),
        reference=ref_loss.item(),
        relative=relative,
        tolerance=tolerance,
    )

    opt.zero_grad(set_to_none=True)
    with autocast():
        loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    finite = all(g is not None and torch.isfinite(g).all().item() for g in grads)
    nonzero = any(g is not None and g.abs().sum().item() > 0 for g in grads)
    d62.add_check(
        report,
        "backward-grads",
        "passed" if finite and nonzero else "failed",
        finite=finite,
        nonzero=nonzero,
        tensors=len(grads),
    )

    before = d62.state_bytes(model, opt)
    opt.step()
    after = d62.state_bytes(model, opt)
    d62.add_check(report, "optimizer-step", "passed" if before != after else "failed")

    opt.zero_grad(set_to_none=True)
    pre_state = d62.state_bytes(model, opt)
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
                step_loss = loss_fn(name, model, inputs)
                step_loss.backward()
            opt.step()
            ref_losses.append(step_loss.item())
        ref_state = d62.state_bytes(model, opt)

        fresh = build_model(name, dtype, device)
        fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
        payload = torch.load(checkpoint, weights_only=True, map_location="cpu")
        fresh.load_state_dict(payload["model"])
        fresh_opt.load_state_dict(payload["opt"])
        torch.set_rng_state(payload["rng"])
        torch.cuda.set_rng_state(payload["cuda_rng"])
        restored = d62.state_bytes(fresh, fresh_opt) == pre_state
        d62.add_check(
            report, "resume-bitwise-state", "passed" if restored else "failed"
        )
        resumed_losses = []
        for _ in range(3):
            fresh_opt.zero_grad(set_to_none=True)
            with autocast():
                step_loss = loss_fn(name, fresh, inputs)
                step_loss.backward()
            fresh_opt.step()
            resumed_losses.append(step_loss.item())
        exact = (
            resumed_losses == ref_losses
            and d62.state_bytes(fresh, fresh_opt) == ref_state
        )
        d62.add_check(
            report,
            "resume-exact-continuation",
            "passed" if exact else "failed",
            reference=ref_losses,
            resumed=resumed_losses,
        )

    exercised = model_routes(name, model)
    for route in spec["routes"]:
        fired = route in exercised and forward_ok and finite and nonzero
        report["routes"][route] = d62.route_status(fired, False, False)
        d62.add_check(
            report, f"route-{route}", "passed" if fired else "failed", fired=fired
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY), required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = "cuda"
    report = d62.new_report(args.model, args.dtype, "native")
    report["versions"] = d62.record_versions()
    violations = d62.owned_import_guard(sys.path)
    if violations:
        d62.add_check(report, "owned-import-guard", "failed", violations=violations)
    else:
        d62.add_check(report, "owned-import-guard", "passed")
    arch = torch.cuda.get_device_properties(0).gcnArchName
    d62.add_check(
        report,
        "gfx1013-native",
        "passed" if arch.split(":")[0] == "gfx1013" else "failed",
        arch=arch,
    )
    admission_battery(report, args.model, args.dtype, device)
    report["status"] = d62.summarize(report)
    passed = sum(
        c["status"] in ("passed", "refused-expected", "absent")
        for c in report["checks"]
    )
    report["summary"] = {
        "total": len(report["checks"]),
        "passed": passed,
        "all_passed": report["status"] == "passed",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"PLAN2_06_ADMIT {args.model}/{args.dtype} status={report['status']} "
        f"routes={report['routes']}",
        flush=True,
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
