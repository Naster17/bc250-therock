#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit native general sparse dispatch and differentiable GPU densification."""

import argparse
import json
from pathlib import Path
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
torch.set_num_threads(4)
report = dict(status="running", checks=[], results=[])
try:
    for dtype in (torch.float32, torch.float64, torch.complex64, torch.complex128):
        dense = torch.tensor([[3, 0, 1], [2, 4, 0], [0, -1, 5]], dtype=dtype)
        csr = dense.to_sparse_csr()
        values = csr.values().cuda().detach().requires_grad_()
        a = torch.sparse_csr_tensor(
            csr.crow_indices().cuda(),
            csr.col_indices().cuda(),
            values.detach(),
            size=(3, 3),
        )
        b = torch.tensor([1, 2, 3], dtype=dtype, device="cuda", requires_grad=True)
        try:
            torch.sparse.spsolve(a, b)
        except RuntimeError as error:
            assert "not supported in ROCm" in str(error)
            report["results"].append(dict(dtype=str(dtype), native_error=str(error)))
        else:
            raise AssertionError("Native API unexpectedly accepted")
        report["checks"].append("native-rejection-" + str(dtype))
        rows, cols = dense.nonzero(as_tuple=True)
        dense_gpu = torch.zeros((3, 3), device="cuda", dtype=dtype).index_put(
            (rows.cuda(), cols.cuda()), values
        )
        x = torch.linalg.solve(dense_gpu, b)
        gradients = torch.autograd.grad(x.abs().square().sum(), (values, b))
        ref_a = dense.to(
            torch.complex128 if dense.is_complex() else torch.float64
        ).requires_grad_()
        ref_b = b.detach().cpu().to(ref_a.dtype).requires_grad_()
        ref_x = torch.linalg.solve(ref_a, ref_b)
        ga, gb = torch.autograd.grad(ref_x.abs().square().sum(), (ref_a, ref_b))
        for actual, expected in (
            (x, ref_x),
            (gradients[0], ga[dense != 0]),
            (gradients[1], gb),
        ):
            torch.testing.assert_close(
                actual.detach().cpu().to(expected.dtype), expected, rtol=2e-5, atol=2e-6
            )
            assert actual.is_cuda
        report["checks"].append("dense-gpu-values-rhs-gradient-" + str(dtype))
    torch.cuda.synchronize()
    report["status"] = "passed"
except BaseException as error:
    report.update(status="failed", error=repr(error))
    raise
finally:
    args.output.write_text(json.dumps(report, indent=2) + "\n")
