#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded Torch convolution regression for the A01 runtime/library changes."""

import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(251019)
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1013"
    cases = [
        dict(
            rank=r,
            groups=g,
            dtype=d,
            gradients="both",
            layout="contiguous",
            transpose=False,
            benchmark=False,
            deterministic=False,
        )
        for r in [1, 2, 3]
        for g in [1, 2]
        for d in ["f32", "f16", "bf16"]
    ]
    base = dict(cases[0])
    for gradients in ["none", "dx", "dw"]:
        cases.append(dict(base, gradients=gradients))
    for dtype in ["f32", "bf16"]:
        cases.append(dict(base, rank=2, layout="channels_last", dtype=dtype))
    cases.append(dict(base, rank=2, layout="strided"))
    cases += [dict(base, benchmark=True), dict(base, deterministic=True)]
    cases += [
        dict(base, rank=2, transpose=True, groups=g, dtype=d)
        for g in [1, 2]
        for d in ["f32", "bf16"]
    ]
    rows = []
    for config in cases:
        rank, groups = config["rank"], config["groups"]
        dtype = dict(f32=torch.float32, f16=torch.float16, bf16=torch.bfloat16)[
            config["dtype"]
        ]
        tol = dict(f32=4e-4, f16=0.003, bf16=0.025)[config["dtype"]]
        shape = {1: (19,), 2: (5, 7), 3: (3, 5, 7)}[rank]
        x = (torch.randn(2, 4, *shape) * 0.2).to(dtype)
        wshape = (4, 6 // groups) if config["transpose"] else (6, 4 // groups)
        w = (torch.randn(*wshape, *((3,) * rank)) * 0.2).to(dtype)
        dx, dw = config["gradients"] in ["both", "dx"], config["gradients"] in [
            "both",
            "dw",
        ]
        cpu = [x.double().requires_grad_(dx), w.double().requires_grad_(dw)]
        gx = x.cuda()
        if config["layout"] == "channels_last":
            gx = gx.contiguous(memory_format=torch.channels_last)
        elif config["layout"] == "strided":
            storage = torch.empty(
                *gx.shape[:-1], gx.shape[-1] * 2, device="cuda", dtype=dtype
            )
            storage[..., ::2].copy_(gx)
            gx = storage[..., ::2]
        gpu = [gx.detach().requires_grad_(dx), w.cuda().requires_grad_(dw)]
        op = getattr(
            F, f"conv_transpose{rank}d" if config["transpose"] else f"conv{rank}d"
        )
        kw = dict(stride=2, padding=2, dilation=2, groups=groups)
        with torch.backends.cudnn.flags(
            benchmark=config["benchmark"], deterministic=config["deterministic"]
        ):
            ref, actual = op(*cpu, **kw), op(*gpu, **kw)
            torch.testing.assert_close(actual.cpu().double(), ref, atol=tol, rtol=tol)
            if dx or dw:
                grad = torch.linspace(-0.4, 0.6, ref.numel()).reshape(ref.shape)
                ref.backward(grad.double())
                actual.backward(grad.to("cuda", dtype))
                for i, required in enumerate([dx, dw]):
                    if required:
                        torch.testing.assert_close(
                            gpu[i].grad.cpu().double(), cpu[i].grad, atol=tol, rtol=tol
                        )
        torch.cuda.synchronize()
        rows.append(dict(config, status="passed"))
        args.output.write_text(
            json.dumps(dict(status="running", results=rows), indent=2) + "\n"
        )
        print(json.dumps(rows[-1]), flush=True)
    args.output.write_text(
        json.dumps(dict(status="passed", results=rows), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
