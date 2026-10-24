#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure strict W8 validation/native components before choosing an optimization."""

import argparse
import json
from pathlib import Path
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import bc250_quant_b051 as quant

    torch.set_num_threads(1)
    torch.manual_seed(540)
    report = dict(status="running", checks=[], shapes=[])
    try:
        for m, n, k in ((8, 128, 256), (32, 256, 256), (8, 512, 1024)):
            x = torch.randn(m, k, device="cuda", requires_grad=True)
            p = quant.pack(torch.randn(n, k, device="cuda"))
            up = torch.randn(m, n, device="cuda")

            def valid_scales():
                return bool(torch.isfinite(p.scales).all() & (p.scales > 0).all())

            def valid_inputs():
                return bool(torch.isfinite(x).all())

            def native_forward():
                return torch._weight_int8pack_mm(x.detach(), p.values, p.scales)

            def native_backward():
                return torch._weight_int8pack_mm(
                    (up.float() * p.scales).contiguous(),
                    p.values.T.contiguous(),
                    torch.ones(k, device="cuda"),
                )

            def strict_forward():
                return quant.linear(x, p)

            def strict_forward_backward():
                x.grad = None
                quant.linear(x, p).backward(up)

            functions = dict(
                scale_check=valid_scales,
                input_check=valid_inputs,
                native_forward=native_forward,
                native_backward=native_backward,
                strict_forward=strict_forward,
                strict_forward_backward=strict_forward_backward,
            )
            measurements = {}
            for name, fn in functions.items():
                fn()
                fn()
                torch.cuda.synchronize()
                samples = []
                for _ in range(3):
                    start = time.perf_counter()
                    for _ in range(5):
                        fn()
                    torch.cuda.synchronize()
                    samples.append((time.perf_counter() - start) * 1000 / 5)
                measurements[name] = dict(
                    samples_ms=samples, median_ms=statistics.median(samples)
                )
            reference = (
                x.detach().cpu().double()
                @ (p.values.cpu().double() * p.scales.cpu().double()[:, None]).T
            )
            torch.testing.assert_close(
                strict_forward().detach().cpu(), reference.float(), rtol=5e-5, atol=2e-5
            )
            strict_forward_backward()
            gradient = (
                up.cpu().double()
                @ (p.values.cpu().double() * p.scales.cpu().double()[:, None])
            ).float()
            torch.testing.assert_close(x.grad.cpu(), gradient, rtol=5e-5, atol=2e-5)
            report["shapes"].append(
                dict(
                    shape=[m, n, k],
                    measurements=measurements,
                    packed_payload_bytes=n * k + 4 * n,
                )
            )
            report["checks"].append(
                "profile-components-cpu-output-gradient-" + str([m, n, k])
            )
        # Separate from latency samples: profiler overhead must not become timing data.
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            with torch.profiler.record_function("strict-w8-forward-backward"):
                strict_forward_backward()
                torch.cuda.synchronize()
        report["operators"] = [
            dict(
                name=e.key,
                count=e.count,
                self_cpu_us=e.self_cpu_time_total,
                device_us=getattr(e, "device_time_total", 0),
            )
            for e in prof.key_averages()
        ]
        report["scalar_reads"] = sum(
            e.count for e in prof.key_averages() if e.key == "aten::_local_scalar_dense"
        )
        assert report["scalar_reads"] == 2
        report["checks"].append("two-strict-validation-scalar-reads")
        prof.export_chrome_trace(str(args.output.parent / "strict-baseline-trace.json"))
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
