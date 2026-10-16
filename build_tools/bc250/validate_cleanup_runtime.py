#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Short CPU-reference gate before/after storage cleanup; no user checkpoints."""

import argparse
import importlib.metadata
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(251016)
    props = torch.cuda.get_device_properties(0)
    assert props.gcnArchName.split(":")[0] == "gfx1013"
    rows = []

    def check(name, op, inputs, tol=4e-4):
        cpu = [x.double().detach().requires_grad_() for x in inputs]
        gpu = [x.cuda().detach().requires_grad_() for x in inputs]
        ref = op(*cpu)
        actual = op(*gpu)
        torch.testing.assert_close(actual.cpu().double(), ref, atol=tol, rtol=tol)
        g = torch.linspace(-0.4, 0.6, ref.numel()).reshape(ref.shape)
        ref.backward(g.double())
        actual.backward(g.to(actual.device, actual.dtype))
        for a, b in zip(gpu, cpu):
            torch.testing.assert_close(
                a.grad.cpu().double(), b.grad, atol=tol, rtol=tol
            )
        rows.append(name)

    for dtype, tol in [(torch.float32, 4e-4), (torch.bfloat16, 0.025)]:
        check(
            "gemm-output-grad-" + str(dtype),
            lambda a, b: a @ b,
            [
                (torch.randn(13, 17) * 0.2).to(dtype),
                (torch.randn(17, 9) * 0.2).to(dtype),
            ],
            tol,
        )
    for d in (32, 256):
        q = torch.randn(1, 2, 17, d) * 0.2
        with sdpa_kernel(SDPBackend.MATH):
            check(
                "math-sdpa-output-qkv-grad-" + str(d),
                lambda a, b, c: F.scaled_dot_product_attention(a, b, c, is_causal=True),
                [q, torch.randn_like(q) * 0.2, torch.randn_like(q) * 0.2],
            )
    check(
        "layernorm-output-grad",
        lambda x: F.layer_norm(x, (17,)),
        [torch.randn(2, 3, 17)],
    )
    check(
        "fft-output-grad",
        lambda x: torch.fft.rfft(x).abs().square().mean(),
        [torch.randn(2, 32) * 0.2],
    )
    torch.cuda.manual_seed_all(210)
    state = torch.cuda.get_rng_state()
    a = torch.randn(101, device="cuda")
    torch.cuda.set_rng_state(state)
    assert torch.equal(a, torch.randn(101, device="cuda"))
    rows.append("rng-exact-replay")
    # The retained consumer bundle must still resolve without reinstalling packages.
    import transformers
    import peft
    import triton

    assert transformers.__version__ == "5.2.0" and peft.__version__ == "0.18.1"
    torch.cuda.synchronize()
    r = {
        "status": "passed",
        "arch": props.gcnArchName,
        "checks": rows,
        "torch_path": torch.__file__,
        "transformers_path": transformers.__file__,
        "versions": {
            k: importlib.metadata.version(k)
            for k in ["torch", "triton", "transformers", "peft", "accelerate"]
        },
    }
    args.output.write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r, indent=2), flush=True)


if __name__ == "__main__":
    main()
