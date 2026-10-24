#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded runtime audit of existing weight-only primitives; no package install."""

import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import triton

    torch.set_num_threads(1)
    assert torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1013:")
    rows = []
    report = dict(
        status="running",
        checks=rows,
        torch=torch.__version__,
        hip=torch.version.hip,
        triton=triton.__version__,
        packages={
            n: importlib.util.find_spec(n) is not None
            for n in ("torchao", "bitsandbytes")
        },
    )
    try:
        q = ((torch.arange(35).reshape(5, 7) % 7) - 3).to(torch.int8)
        scale = torch.arange(1, 6, dtype=torch.float32) / 8
        x = (torch.arange(21).reshape(3, 7).float() - 10) / 8
        reference = x.double() @ (q.double() * scale.double()[:, None]).T
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            actual = torch._weight_int8pack_mm(
                x.cuda().to(dtype), q.cuda(), scale.cuda()
            )
            torch.testing.assert_close(
                actual.cpu(), reference.to(dtype), rtol=0, atol=0
            )
            rows.append(
                dict(name="existing-native-weight-int8-" + str(dtype), status="passed")
            )
        noncontiguous = (
            torch.arange(21, dtype=torch.float32)
            .reshape(7, 3)
            .T.cuda()
            .T.contiguous()
            .T
        )
        # Preserve the transpose layout explicitly on device.
        assert noncontiguous.shape == (3, 7) and not noncontiguous.is_contiguous()
        actual = torch._weight_int8pack_mm(noncontiguous, q.cuda(), scale.cuda())
        expected = (
            noncontiguous.cpu().double() @ (q.double() * scale.double()[:, None]).T
        ).float()
        mismatch = int((actual.cpu() != expected).sum())
        assert mismatch > 0, "Source-audited stride bug no longer reproduces; re-audit"
        rows.append(
            dict(
                name="native-strided-input-defect",
                status="reproduced",
                mismatched_elements=mismatch,
                actual=actual.cpu().tolist(),
                expected=expected.tolist(),
            )
        )
        scales_storage = torch.zeros(10, device="cuda")
        scales_storage[::2] = scale.cuda()
        actual = torch._weight_int8pack_mm(x.cuda(), q.cuda(), scales_storage[::2])
        mismatch = int((actual.cpu() != reference.float()).sum())
        assert mismatch > 0, (
            "Source-audited scale stride bug no longer reproduces; re-audit"
        )
        rows.append(
            dict(
                name="native-strided-scale-defect",
                status="reproduced",
                mismatched_elements=mismatch,
            )
        )
        value = x.cuda().requires_grad_()
        output = torch._weight_int8pack_mm(value, q.cuda(), scale.cuda())
        try:
            output.sum().backward()
        except RuntimeError as error:
            assert "derivative for aten::_weight_int8pack_mm is not implemented" in str(
                error
            )
            assert value.grad is None
            rows.append(
                dict(
                    name="native-input-gradient-unavailable",
                    status="expected-refusal",
                    error=str(error),
                )
            )
        else:
            raise AssertionError("Native backward policy changed; re-audit")
        try:
            torch._convert_weight_to_int4pack(
                torch.zeros((16, 64), device="cuda", dtype=torch.uint8), 2
            )
        except RuntimeError as error:
            assert "CDNA2" in str(error), error
            rows.append(
                dict(
                    name="native-packed-int4-gfx1013-refusal",
                    status="expected-refusal",
                    error=str(error),
                )
            )
        else:
            raise AssertionError("Native INT4 admission changed; re-audit")
        torch.cuda.synchronize()
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
