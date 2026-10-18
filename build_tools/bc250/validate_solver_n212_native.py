#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Private N2.12 ABI refusal and sentinel-protected nondefault-stream outputs."""

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    import torch
    import gfx1013_schur_candidate as c

    torch.set_num_threads(4)
    torch.manual_seed(10132120)
    report = dict(status="running", cases=[], refusals=[])
    for dtype in (torch.complex64, torch.complex128):
        for n in (1, 2, 16):
            t = 0.3 * torch.triu(torch.randn(n, n, dtype=dtype, device="cuda"))
            t.diagonal().copy_(torch.linspace(-0.8, 0.8, n, device="cuda") + 0.1j)
            before = t.clone()
            # Flat contiguous output region, surrounded by exact sentinels.
            storage = torch.full((n * n + 16,), 823 - 519j, dtype=dtype, device="cuda")
            y = storage[8:-8].view(n, n)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                expected = c._backsubstitution_reference(t)
                c._backsubstitution_device(t)  # bind the private ABI normally
                abi = c._BACKSUB_LIBRARY.bc250_triangular_backsub
                code = abi(
                    t.data_ptr(),
                    y.data_ptr(),
                    n,
                    int(dtype == torch.complex128),
                    stream.cuda_stream,
                )
                assert code == 0, code
            stream.synchronize()
            tol = 8e-5 if dtype == torch.complex64 else 8e-13
            torch.testing.assert_close(y, expected, rtol=tol, atol=tol)
            torch.testing.assert_close(
                storage[:8], torch.full_like(storage[:8], 823 - 519j), rtol=0, atol=0
            )
            torch.testing.assert_close(
                storage[-8:], torch.full_like(storage[-8:], 823 - 519j), rtol=0, atol=0
            )
            torch.testing.assert_close(t, before, rtol=0, atol=0)
            report["cases"].append(
                dict(
                    dtype=str(dtype),
                    order=n,
                    guarded_output=True,
                    input_unchanged=True,
                    nondefault_stream=True,
                )
            )
    for label, ptr, out, n, fp64 in (
        ("null-input", None, y.data_ptr(), 16, 1),
        ("null-output", t.data_ptr(), None, 16, 1),
        ("exact-alias", t.data_ptr(), t.data_ptr(), 16, 1),
        ("zero-order", t.data_ptr(), y.data_ptr(), 0, 1),
        ("oversize", t.data_ptr(), y.data_ptr(), 17, 1),
        ("invalid-precision", t.data_ptr(), y.data_ptr(), 16, 2),
    ):
        code = abi(ptr, out, n, fp64, torch.cuda.current_stream().cuda_stream)
        assert code == 1, (label, code)  # hipErrorInvalidValue, before any launch
        report["refusals"].append(label)
    torch.cuda.synchronize()
    report["status"] = "passed"
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
