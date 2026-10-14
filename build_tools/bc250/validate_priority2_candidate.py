#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Bounded immutable-prefix regression for the earlier sparse-boundary crash."""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import time

import torch


def main() -> None:
    faulthandler.enable(all_threads=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1013"
    assert os.geteuid() == 1000
    assert torch.__file__.startswith("/workspace/pytorch-work/torch/")
    torch.set_num_threads(4)
    torch.manual_seed(2502)
    started = time.monotonic()
    cpu = torch.randn(31, 29, dtype=torch.float64)
    cpu[cpu.abs() < 0.9] = 0
    rhs = torch.randn(29, 13, dtype=torch.float64)
    expected = cpu @ rhs
    gpu_rhs = rhs.cuda()
    compiled = torch.compile(lambda a: a.sin() + a.square(), fullgraph=True)
    x = torch.randn(129, device="cuda", requires_grad=True)
    expected_forward = x.detach().cpu().sin() + x.detach().cpu().square()
    with torch.sparse.check_sparse_tensor_invariants():
        for cycle in range(20):
            for layout in (torch.sparse_coo, torch.sparse_csr):
                sparse = cpu.to_sparse(layout=layout).cuda()
                result = torch.sparse.mm(sparse, gpu_rhs)
                torch.testing.assert_close(
                    result.cpu(), expected, rtol=1e-10, atol=1e-10
                )
            f = compiled(x)
            torch.testing.assert_close(f.cpu(), expected_forward)
            x.grad = None
            f.sum().backward()
            torch.testing.assert_close(
                x.grad.cpu(), x.detach().cpu().cos() + 2 * x.detach().cpu()
            )
            # Alternating allocation/copy/free and event ordering, bounded shared RAM use.
            temporary = torch.full(
                (16 * 1024 * 1024,), cycle, dtype=torch.uint8, device="cuda"
            )
            assert temporary[-1].item() == cycle
            del temporary
            frequency = torch.fft.rfft(x.detach())
            torch.testing.assert_close(
                torch.fft.irfft(frequency, n=x.numel()).cpu(), x.detach().cpu()
            )
    checkpoint = args.output.with_suffix(".pt")
    torch.save({"sparse_result": result, "gradient": x.grad}, checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    torch.testing.assert_close(state["sparse_result"], expected, rtol=1e-10, atol=1e-10)
    report = {
        "status": "passed",
        "cycles": 20,
        "sparse_matmuls": 40,
        "compiled_forward_backward": 20,
        "arch": torch.cuda.get_device_properties(0).gcnArchName,
        "torch": torch.__version__,
        "torch_file": torch.__file__,
        "seconds": time.monotonic() - started,
        "note": "No concurrent prefix writes; does not establish the historical crash root cause.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("PRIORITY2_CANDIDATE_PASS", flush=True)


if __name__ == "__main__":
    main()
