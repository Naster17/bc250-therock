#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CPU-reference ordinary Torch convolution forward and all first derivatives."""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--sha", required=True)
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    assert hashlib.sha256(a.library.read_bytes()).hexdigest() == a.sha
    if not a.worker:
        env = dict(
            os.environ,
            LD_LIBRARY_PATH=str(a.library.parent)
            + ":/workspace/build/dist/rocm/lib/rocm_sysdeps/lib:"
            + os.environ.get("LD_LIBRARY_PATH", ""),
        )
        subprocess.run(
            [sys.executable, __file__, *sys.argv[1:], "--worker"],
            env=env,
            check=True,
            timeout=430,
        )
        return
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(4)
    C.CDLL("libMIOpen.so.1")
    assert str(a.library.resolve()) in Path("/proc/self/maps").read_text()
    r = dict(status="running", checks=[], library_sha256=a.sha)
    try:
        assert torch.cuda.get_device_properties(0).gcnArchName == "gfx1013:xnack-"
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            for groups in [1, 3]:
                for dilation in [1, 2]:
                    torch.manual_seed(1711)
                    values = [
                        torch.randn(2, 4 * groups, 9, 11) / 8,
                        torch.randn(8 * groups, 4, 3, 3) / 8,
                        torch.randn(8 * groups) / 8,
                    ]
                    values = [v.to(dtype) for v in values]
                    cpu = [v.double().requires_grad_() for v in values]
                    gpu = [v.cuda().requires_grad_() for v in values]
                    ref = F.conv2d(
                        *cpu, padding=dilation, dilation=dilation, groups=groups
                    )
                    actual = F.conv2d(
                        *gpu, padding=dilation, dilation=dilation, groups=groups
                    )
                    cot = (
                        torch.linspace(-0.5, 0.5, ref.numel()).reshape_as(ref).to(dtype)
                    )
                    (ref * cot.double()).sum().backward()
                    (actual * cot.cuda()).sum().backward()
                    bound = (
                        (0.002, 0.003, 0.003, 0.006)
                        if dtype == torch.float16
                        else (
                            (0.003, 0.01, 0.01, 0.02)
                            if dtype == torch.bfloat16
                            else (3e-6, 5e-5, 3e-6, 5e-5)
                        )
                    )
                    torch.testing.assert_close(
                        actual.cpu().double(), ref, atol=bound[0], rtol=bound[1]
                    )
                    for x, y in zip(cpu, gpu):
                        torch.testing.assert_close(
                            y.grad.cpu().double(), x.grad, atol=bound[2], rtol=bound[3]
                        )
                    r["checks"].append(
                        f"{dtype}-g{groups}-dilation{dilation}-output-dx-dw-db"
                    )
        torch.cuda.synchronize()
        r["status"] = "passed"
    except BaseException as e:
        r.update(status="failed", error=repr(e))
        raise
    finally:
        a.output.write_text(json.dumps(r, indent=2) + "\n")


if __name__ == "__main__":
    main()
