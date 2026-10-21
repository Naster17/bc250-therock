#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Probe current BF16 FFT dispatch and explicit float promotion before coding."""

import argparse
import json
from pathlib import Path
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(4)
x = (
    torch.arange(35, dtype=torch.float32, device="cuda")
    .reshape(5, 7)
    .div(41)
    .to(torch.bfloat16)
    .requires_grad_()
)
report = dict(status="running", checks=[], results=[], torch=str(torch.__version__))
for name, context in (
    ("outside-autocast", torch.autocast("cuda", enabled=False)),
    ("bf16-autocast", torch.autocast("cuda", dtype=torch.bfloat16)),
):
    with context:
        try:
            y = torch.fft.rfft2(x)
        except RuntimeError as error:
            report["results"].append(
                dict(name=name, outcome="rejected", error=str(error))
            )
        else:
            report["results"].append(
                dict(name=name, outcome="accepted", dtype=str(y.dtype))
            )
    report["checks"].append(name)
y = torch.fft.rfft2(x.float())
g = torch.autograd.grad((y.real.square() + y.imag.square()).sum(), x)[0]
reference = torch.fft.rfft2(x.detach().cpu().double())
torch.testing.assert_close(y.detach().cpu().cdouble(), reference, rtol=5e-4, atol=5e-4)
assert (
    y.dtype == torch.complex64
    and g.dtype == torch.bfloat16
    and g.is_cuda
    and torch.isfinite(g).all()
)
torch.cuda.synchronize()
report["checks"].append("explicit-float-promotion-output-backward")
report.update(status="passed", promoted_dtype=str(y.dtype), gradient_dtype=str(g.dtype))
args.output.write_text(json.dumps(report, indent=2) + "\n")
