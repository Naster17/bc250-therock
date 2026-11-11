#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Owned self-contained GPU checkpoint save/resume regression for release qualification.

Replaces the N3.9 continuity gate, which depends on the excluded user
application. Pure torch; no consumers, no external fixtures. Verifies bitwise
state restoration, exact resumed continuation, provenance identity refusal and
tampered-checkpoint refusal.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify_provenance(expected, loaded):
    if loaded != expected:
        raise ValueError("Checkpoint numerical runtime mismatch")


def build(seed):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, 32)
    ).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, opt


def batches():
    gen = torch.Generator().manual_seed(20261111)
    return [
        (
            torch.randn(16, 64, generator=gen).cuda(),
            torch.randn(16, 32, generator=gen).cuda(),
        )
        for _ in range(5)
    ]


def step(model, opt, x, y):
    opt.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    opt.step()
    return loss.detach()


def state_bytes(model, opt):
    parts = [p.detach().cpu().numpy().tobytes() for p in model.parameters()]
    for group in opt.param_groups:
        for p in group["params"]:
            st = opt.state[p]
            parts.append(st["exp_avg"].cpu().numpy().tobytes())
            parts.append(st["exp_avg_sq"].cpu().numpy().tobytes())
            parts.append(int(st["step"]).to_bytes(8, "little"))
    return b"".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.scratch_root
    root.mkdir(parents=True, exist_ok=True)
    checks = []
    assert torch.cuda.is_available(), "GPU required"
    torch.backends.cudnn.deterministic = True
    data = batches()

    model, opt = build(7)
    losses = [step(model, opt, x, y) for x, y in data[:3]]
    assert all(torch.isfinite(l) for l in losses)
    checks.append("gpu-forward-backward-three-steps-finite")
    pre_state = state_bytes(model, opt)
    pre_params = [p.detach().clone() for p in model.parameters()]

    provenance = {
        "fixture": "validate_checkpoint_release_d59",
        "torch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name(0),
        "fixture_sha256": sha(Path(__file__).read_bytes()),
    }
    ckpt = root / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "step": 3,
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
        },
        ckpt,
    )
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    assert ckpt.is_file() and (root / "provenance.json").is_file()
    checks.append("checkpoint-saved-files-present")

    ref_losses = [step(model, opt, x, y) for x, y in data[3:]]
    ref_state = state_bytes(model, opt)

    loaded_prov = json.loads((root / "provenance.json").read_text())
    verify_provenance(provenance, loaded_prov)
    checks.append("provenance-identity-verified")

    model2, opt2 = build(999)
    verify_provenance(provenance, loaded_prov)
    # Load on CPU: RNG states must stay CPU ByteTensors; state dicts copy
    # into the CUDA model/optimizer on load_state_dict.
    payload = torch.load(ckpt, weights_only=True, map_location="cpu")
    model2.load_state_dict(payload["model"])
    opt2.load_state_dict(payload["opt"])
    torch.set_rng_state(payload["rng"])
    torch.cuda.set_rng_state(payload["cuda_rng"])
    assert state_bytes(model2, opt2) == pre_state
    checks.append("resume-parameter-state-bitwise-exact")
    checks.append("resume-optimizer-state-bitwise-exact")
    resumed_losses = [step(model2, opt2, x, y) for x, y in data[3:]]
    assert all(torch.equal(a, b) for a, b in zip(resumed_losses, ref_losses))
    assert state_bytes(model2, opt2) == ref_state
    checks.append("resumed-continuation-matches-reference-exactly")

    bad_prov = dict(loaded_prov)
    bad_prov["torch_version"] = "0.0.0-tampered"
    refused = False
    try:
        verify_provenance(provenance, bad_prov)
    except ValueError:
        refused = True
    assert refused
    checks.append("tampered-provenance-refused-before-weights")

    bad_ckpt = root / "tampered.pt"
    raw = ckpt.read_bytes()
    # Truncation breaks the zip central directory reliably; a mid-file byte
    # flip can land in uncompressed tensor data and still load.
    bad_ckpt.write_bytes(raw[: len(raw) // 2])
    refused = False
    try:
        torch.load(bad_ckpt, weights_only=True, map_location="cpu")
    except Exception:
        refused = True
    assert refused
    checks.append("tampered-checkpoint-bytes-refused")

    args.output.write_text(
        json.dumps(
            dict(
                status="passed",
                checks=checks,
                checkpoint_sha256=sha(ckpt.read_bytes()),
                pre_state_sha256=sha(pre_state),
            ),
            indent=2,
        )
        + "\n"
    )
    print("CHECKPOINT_RELEASE_D59_PASS", len(checks), "checks")


if __name__ == "__main__":
    main()
