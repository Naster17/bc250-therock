#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Verify user-mode developer GPU execution and a host-visible checkpoint."""
import json
import os
from pathlib import Path

import torch

assert os.geteuid() == 1000
assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1013"
assert torch.backends.cuda.preferred_blas_library() == torch._C._BlasBackend.Cublas
x = torch.linspace(-1, 1, 257, device="cuda", requires_grad=True)
y = torch.compile(lambda a: a.sin() + a.square(), fullgraph=True)(x)
expected = x.detach().cpu().sin() + x.detach().cpu().square()
torch.testing.assert_close(y.cpu(), expected)
y.sum().backward()
torch.testing.assert_close(x.grad.cpu(), x.detach().cpu().cos() + 2 * x.detach().cpu())
path = Path.home() / "bc250work/priority1-dev-checkpoint.pt"
torch.save({"value": y.detach(), "gradient": x.grad}, path)
state = torch.load(path, map_location="cpu", weights_only=True)
torch.testing.assert_close(state["value"], expected)
assert path.stat().st_uid == 1000
report = {
    "status": "passed",
    "uid": os.geteuid(),
    "torch": torch.__version__,
    "checkpoint": str(path),
    "arch": torch.cuda.get_device_properties(0).gcnArchName,
}
(Path.home() / "bc250work/priority1-dev-launcher.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print("PRIORITY1_DEV_LAUNCHER_PASS", flush=True)
