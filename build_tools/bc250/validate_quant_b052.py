#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exact CPU references for native weight-only INT8 layout normalization."""

import argparse
import itertools
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(1)
    report = dict(status="running", checks=[])

    def passed(name):
        report["checks"].append(name)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def matrix(reference, layout):
        actual = reference.cuda()
        if layout == "contiguous":
            return actual, actual
        if layout == "transpose":
            owner = actual.T.contiguous()
            return owner.T, owner
        if layout == "slice":
            owner = torch.full(
                (actual.shape[0] * 2 + 2, actual.shape[1] * 2 + 2),
                19,
                device="cuda",
                dtype=actual.dtype,
            )
            view = owner[
                1 : 1 + 2 * actual.shape[0] : 2, 1 : 1 + 2 * actual.shape[1] : 2
            ]
            view.copy_(actual)
            return view, owner
        owner = actual[:1].clone()
        return owner.expand_as(actual), owner

    try:
        assert torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1013:")
        assert (
            not torch.distributed.is_nccl_available()
            and torch.distributed.is_gloo_available()
        )
        passed("native-device-and-candidate-backend")
        for dtype, (m, n, k), xlayout, slayout, wlayout in itertools.product(
            (torch.float32, torch.float16, torch.bfloat16),
            ((3, 5, 7), (2, 17, 129)),
            ("contiguous", "transpose", "slice", "expand"),
            ("contiguous", "stride", "expand"),
            ("contiguous", "transpose", "slice"),
        ):
            xr = ((torch.arange(m * k).reshape(m, k) % 17) - 8).to(dtype) / 8
            wr = ((torch.arange(n * k).reshape(n, k) % 31) - 15).to(torch.int8)
            sr = (torch.arange(n, dtype=torch.float32) % 5 + 1) / 16
            x, xowner = matrix(xr, xlayout)
            w, wowner = matrix(wr, wlayout)
            if xlayout == "expand":
                xr = xr[:1].expand_as(xr)
            if slayout == "stride":
                sowner = torch.full((2 * n + 4,), 19.0, device="cuda")
                s = sowner[1 : 2 * n + 1 : 2]
                s.copy_(sr)
            elif slayout == "expand":
                sowner = sr[:1].cuda()
                s = sowner.expand(n)
                sr = sr[:1].expand(n)
            else:
                s = sr.cuda()
                sowner = s
            originals = [p.clone() for p in (xowner, wowner, sowner)]
            expected = ((xr.double() @ wr.double().T) * sr.double()).to(dtype)
            actual = torch._weight_int8pack_mm(x, w, s)
            assert (
                actual.device == x.device
                and actual.dtype == dtype
                and actual.shape == (m, n)
            )
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
            for owner, original in zip((xowner, wowner, sowner), originals):
                torch.testing.assert_close(owner, original, rtol=0, atol=0)
            passed(f"exact-guards-{dtype}-{m}-{n}-{k}-{xlayout}-{slayout}-{wlayout}")
        # Independent odd/tail transpose proof on a private stream.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            x = torch.arange(21, device="cuda", dtype=torch.float32).reshape(7, 3).T
            w = torch.arange(35, device="cuda", dtype=torch.int8).reshape(7, 5).T
            storage = torch.full((10,), 19.0, device="cuda")
            storage[::2] = torch.arange(1, 6, device="cuda") / 8
            expected = (x.cpu().double() @ w.cpu().double().T) * storage[
                ::2
            ].cpu().double()
            actual = torch._weight_int8pack_mm(x, w, storage[::2])
            torch.testing.assert_close(actual.cpu(), expected.float(), rtol=0, atol=0)
        torch.cuda.current_stream().wait_stream(stream)
        passed("private-stream-three-strided-operands")
        x = torch.ones((2, 7), device="cuda", requires_grad=True)
        try:
            torch._weight_int8pack_mm(
                x,
                torch.ones((5, 7), device="cuda", dtype=torch.int8),
                torch.ones(5, device="cuda"),
            ).sum().backward()
        except RuntimeError as error:
            assert "derivative for aten::_weight_int8pack_mm is not implemented" in str(
                error
            )
            assert x.grad is None
            report["backward_refusal"] = str(error)
        else:
            raise AssertionError("Native backward boundary unexpectedly changed")
        passed("native-backward-still-refuses")
        try:
            torch._convert_weight_to_int4pack(
                torch.zeros((16, 64), device="cuda", dtype=torch.uint8), 2
            )
        except RuntimeError as error:
            assert "CDNA2" in str(error)
            report["int4_refusal"] = str(error)
        else:
            raise AssertionError("INT4 architecture admission unexpectedly changed")
        passed("native-int4-still-refuses-gfx1013")
        torch.cuda.synchronize()
        assert len(report["checks"]) == 220
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
