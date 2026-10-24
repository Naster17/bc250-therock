#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Independent CPU references for the isolated frozen W8 linear recipe."""

import argparse
import json
from pathlib import Path
import struct
import time


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def reference_pack(source, torch):
    values, scales = [], []
    for row in source.float().tolist():
        maximum = max(abs(v) for v in row)
        scale = f32(maximum * f32(1.0 / 127.0)) if maximum else 1.0
        scales.append(scale)
        values.append([max(-127, min(127, round(f32(v / scale)))) for v in row])
    return torch.tensor(values, dtype=torch.int8), torch.tensor(
        scales, dtype=torch.float32
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import bc250_quant_b051 as quant

    torch.set_num_threads(1)
    torch.manual_seed(511)
    report = dict(status="running", checks=[], errors=[], timings=[])

    def passed(name):
        report["checks"].append(name)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def exact(a, b):
        torch.testing.assert_close(a.cpu(), b, rtol=0, atol=0)

    def refuses(name, operation):
        try:
            operation()
        except ValueError as error:
            passed("refuse-" + name)
            report["errors"].append(dict(name=name, error=str(error)))
        else:
            raise AssertionError("Missing refusal: " + name)

    try:
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            tol = (
                {"rtol": 5e-5, "atol": 3e-6}
                if dtype == torch.float32
                else {"rtol": 3e-3, "atol": 5e-4}
                if dtype == torch.float16
                else {"rtol": 2e-2, "atol": 4e-3}
            )
            for m, n, k in ((1, 5, 7), (3, 9, 17), (6, 13, 64), (2, 17, 129)):
                source = torch.randn(n, k).to(dtype)
                source[0].zero_()
                qref, sref = reference_pack(source, torch)
                wref = qref.double() * sref.double()[:, None]
                for layout in ("contiguous", "strided", "batched"):
                    weight_source = source.cuda()
                    if layout == "strided":
                        weight_source = weight_source.T.contiguous().T
                    packed = quant.pack(weight_source)
                    exact(packed.values, qref)
                    exact(packed.scales, sref)
                    values_before = packed.values.clone()
                    scales_before = packed.scales.clone()
                    if layout == "strided":
                        scale_storage = torch.full((2 * n + 3,), 19.0, device="cuda")
                        scale_view = scale_storage[1 : 2 * n + 1 : 2]
                        scale_view.copy_(packed.scales)
                        packed = quant.FrozenInt8Weight(packed.values, scale_view)
                    xref = torch.randn(m, k).to(dtype) / 4
                    upstream = (torch.randn(m, n) / 4).to(dtype)
                    if layout == "strided":
                        storage = torch.full(
                            (m, 2 * k + 3), 19, device="cuda", dtype=dtype
                        )
                        x = storage[:, 1 : 2 * k + 1 : 2]
                        x.copy_(xref)
                        x.requires_grad_()
                        before = storage.clone()
                    else:
                        x = xref.cuda()
                        if layout == "batched":
                            x = x.reshape(1, m, k)
                        x.requires_grad_()
                    bias = (torch.randn(n) / 8).to(dtype).cuda().requires_grad_()
                    y = quant.linear(x, packed, bias)
                    y.backward(upstream.cuda().reshape_as(y))
                    expected = (xref.double() @ wref.T).to(dtype) + bias.detach().cpu()
                    torch.testing.assert_close(
                        y.detach().reshape(m, n).cpu(), expected, **tol
                    )
                    dx = (upstream.double() @ wref).to(dtype)
                    torch.testing.assert_close(x.grad.reshape(m, k).cpu(), dx, **tol)
                    torch.testing.assert_close(
                        bias.grad.cpu(), upstream.float().sum(0).to(dtype), **tol
                    )
                    exact(packed.values, values_before.cpu())
                    exact(packed.scales, scales_before.cpu())
                    if layout == "strided":
                        exact(storage, before.cpu())
                        exact(scale_storage[0:1], torch.tensor([19.0]))
                        exact(scale_storage[2 : 2 * n + 1 : 2], torch.full((n,), 19.0))
                    expected_bytes = n * k + 4 * n
                    assert (
                        packed.values.numel() * packed.values.element_size()
                        + packed.scales.numel() * packed.scales.element_size()
                        == expected_bytes
                    )
                    passed(
                        f"pack-output-input-bias-grad-guards-{dtype}-{m}-{n}-{k}-{layout}"
                    )
        # Explicit ties, zero row, symmetric endpoints and all signed byte values.
        source = torch.tensor(
            [[0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 127.0, -127.0], [0.0] * 8], device="cuda"
        )
        p = quant.pack(source)
        exact(
            p.values,
            torch.tensor([[0, 2, 2, 0, -2, -2, 127, -127], [0] * 8], dtype=torch.int8),
        )
        exact(p.scales, torch.ones(2))
        passed("round-even-zero-row-symmetric-endpoints")
        values = (
            torch.arange(-128, 128, dtype=torch.int16).to(torch.int8).reshape(16, 16)
        )
        p = quant.FrozenInt8Weight(values.cuda(), torch.ones(16, device="cuda") / 8)
        x = (torch.arange(32).reshape(2, 16).float() / 8).cuda().requires_grad_()
        y = quant.linear(x, p)
        y.sum().backward()
        exact(y, (x.detach().cpu().double() @ (values.double() / 8).T).float())
        exact(x.grad, (values.double() / 8).sum(0).repeat(2, 1).float())
        passed("all256-signed-byte-values-output-gradient")
        # Nondefault stream with explicit handoff, and a finite-difference ladder.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            x = torch.arange(6, device="cuda", dtype=torch.float32).reshape(2, 3) / 8
            p = quant.FrozenInt8Weight(
                torch.tensor([[1, -2, 3], [-3, 1, 2]], device="cuda", dtype=torch.int8),
                torch.tensor([0.125, 0.25], device="cuda"),
            )
            x.requires_grad_()
            u = torch.tensor([[0.25, -0.5], [0.5, 0.25]], device="cuda")
            (quant.linear(x, p) * u).sum().backward()
            numerical = []
            for i in range(6):
                plus = x.detach().clone()
                minus = x.detach().clone()
                plus.view(-1)[i] += 1 / 64
                minus.view(-1)[i] -= 1 / 64
                numerical.append(
                    (
                        (quant.linear(plus, p) * u).sum()
                        - (quant.linear(minus, p) * u).sum()
                    )
                    * 32
                )
            exact(torch.stack(numerical).reshape(2, 3), x.grad.cpu())
        torch.cuda.current_stream().wait_stream(stream)
        passed("private-stream-handoff-and-six-exact-finite-differences")
        # Owned adapter composite: frozen quantized base plus trainable low-rank branch.
        for dtype in (torch.float32, torch.bfloat16):
            source = ((torch.arange(63).reshape(7, 9) % 127) - 63).to(dtype) / 128
            source[:, -1] = 127 / 128
            p = quant.pack(source.cuda())
            q, s = reference_pack(source, torch)
            xr = (torch.arange(36).reshape(4, 9).float() / 64).to(dtype)
            ar = (torch.arange(18).reshape(9, 2).float() / 128).to(dtype)
            br = (torch.arange(14).reshape(2, 7).float() / 128).to(dtype)
            x = xr.cuda().requires_grad_()
            a = ar.cuda().requires_grad_()
            b = br.cuda().requires_grad_()
            cpu = [t.clone().double().requires_grad_() for t in (xr, ar, br)]
            base = (cpu[0] @ (q.double() * s.double()[:, None]).T).to(dtype)
            hidden = (cpu[0] @ cpu[1]).to(dtype)
            adapter = (hidden.double() @ cpu[2]).to(dtype)
            expected = base + adapter
            y = quant.linear(x, p) + (x @ a) @ b
            exact(y, expected.detach())
            u = torch.full_like(y, 0.125)
            y.backward(u)
            expected.backward(u.cpu())
            tol = (
                dict(rtol=5e-5, atol=3e-6)
                if dtype == torch.float32
                else dict(rtol=2e-2, atol=4e-3)
            )
            for actual, ref in zip((x, a, b), cpu):
                torch.testing.assert_close(actual.grad.cpu(), ref.grad.to(dtype), **tol)
            passed("owned-frozen-base-adapter-all-first-gradients-" + str(dtype))
        # Explicit input dtype is retained under AMP; this is not F.linear AMP routing.
        x = torch.ones((2, 3), device="cuda", requires_grad=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = quant.linear(x, quant.pack(torch.ones((2, 3), device="cuda")))
        assert y.dtype == torch.float32
        y.sum().backward()
        exact(x.grad, torch.full((2, 3), 2.0))
        passed("explicit-input-dtype-under-amp")
        x = torch.ones((2, 3), device="cuda", requires_grad=True)
        p = quant.pack(torch.ones((2, 3), device="cuda"))
        grad = torch.autograd.grad(quant.linear(x, p).sum(), x, create_graph=True)[0]
        assert not grad.requires_grad
        passed("higher-derivatives-not-advertised")
        good = torch.ones((2, 3), device="cuda")
        packed = quant.pack(torch.ones((2, 3), device="cuda"))
        for name, op in (
            (
                "trainable-base",
                lambda: quant.pack(
                    torch.ones((2, 3), device="cuda", requires_grad=True)
                ),
            ),
            ("cpu-base", lambda: quant.pack(torch.ones(2, 3))),
            (
                "double-base",
                lambda: quant.pack(
                    torch.ones((2, 3), device="cuda", dtype=torch.float64)
                ),
            ),
            (
                "nonfinite-base",
                lambda: quant.pack(torch.full((2, 3), float("nan"), device="cuda")),
            ),
            ("rank-base", lambda: quant.pack(torch.ones(3, device="cuda"))),
            ("empty-base", lambda: quant.pack(torch.empty((0, 3), device="cuda"))),
            (
                "oversized-base",
                lambda: quant.pack(torch.ones(1, device="cuda").expand(4096, 4096)),
            ),
            ("cpu-input", lambda: quant.linear(good.cpu(), packed)),
            ("double-input", lambda: quant.linear(good.double(), packed)),
            ("wrong-k", lambda: quant.linear(good[:, :2], packed)),
            ("empty-input", lambda: quant.linear(good[:0], packed)),
            (
                "nonfinite-input",
                lambda: quant.linear(torch.full_like(good, float("inf")), packed),
            ),
            (
                "oversized-input",
                lambda: quant.linear(good[0:1].expand(4097, 3), packed),
            ),
            (
                "format-version",
                lambda: quant.linear(
                    good, quant.FrozenInt8Weight(packed.values, packed.scales, 2)
                ),
            ),
            ("unknown-format", lambda: quant.linear(good, None)),
            (
                "wrong-values-dtype",
                lambda: quant.linear(
                    good, quant.FrozenInt8Weight(packed.values.float(), packed.scales)
                ),
            ),
            (
                "wrong-scale-dtype",
                lambda: quant.linear(
                    good, quant.FrozenInt8Weight(packed.values, packed.scales.half())
                ),
            ),
            (
                "wrong-scale-shape",
                lambda: quant.linear(
                    good, quant.FrozenInt8Weight(packed.values, packed.scales[:, None])
                ),
            ),
            (
                "trainable-scale",
                lambda: quant.linear(
                    good,
                    quant.FrozenInt8Weight(
                        packed.values, packed.scales.clone().requires_grad_()
                    ),
                ),
            ),
            (
                "zero-scale",
                lambda: quant.linear(
                    good,
                    quant.FrozenInt8Weight(
                        packed.values, torch.zeros_like(packed.scales)
                    ),
                ),
            ),
            (
                "negative-scale",
                lambda: quant.linear(
                    good, quant.FrozenInt8Weight(packed.values, -packed.scales)
                ),
            ),
            (
                "nonfinite-scale",
                lambda: quant.linear(
                    good,
                    quant.FrozenInt8Weight(packed.values, packed.scales * float("nan")),
                ),
            ),
            (
                "cpu-values",
                lambda: quant.linear(
                    good, quant.FrozenInt8Weight(packed.values.cpu(), packed.scales)
                ),
            ),
            (
                "bias-shape",
                lambda: quant.linear(good, packed, torch.ones(3, device="cuda")),
            ),
            (
                "bias-dtype",
                lambda: quant.linear(
                    good, packed, torch.ones(2, device="cuda", dtype=torch.float16)
                ),
            ),
        ):
            refuses(name, op)
        # Kernel trace confirms existing native weight-only forward and backward paths.
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            x = good.clone().requires_grad_()
            quant.linear(x, packed).sum().backward()
            torch.cuda.synchronize()
        kernels = [
            e.name for e in prof.events() if "weight_int8pack_mm_kernel" in e.name
        ]
        assert len(kernels) >= 2, kernels
        report["kernel_names"] = kernels
        prof.export_chrome_trace(str(args.output.parent / "quant-native-trace.json"))
        passed("native-weight-only-forward-and-backward-kernel-trace")
        for m, n, k in ((8, 128, 256), (32, 256, 256)):
            x = torch.randn(m, k, device="cuda")
            p = quant.pack(torch.randn(n, k, device="cuda"))
            resident = p.values.float() * p.scales[:, None]

            def fast():
                return quant.linear(x, p)

            def dense():
                return torch.nn.functional.linear(x, resident)

            fast()
            dense()
            torch.cuda.synchronize()
            pairs = []
            for i in range(3):
                samples = {}
                for name, fn in (
                    (("recipe", fast), ("resident-dense", dense))
                    if i % 2 == 0
                    else (("resident-dense", dense), ("recipe", fast))
                ):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    for _ in range(3):
                        fn()
                    torch.cuda.synchronize()
                    samples[name] = (time.perf_counter() - start) * 1000 / 3
                pairs.append(samples)
            report["timings"].append(
                dict(
                    shape=[m, n, k],
                    pairs_ms=pairs,
                    packed_weight_bytes=n * k + 4 * n,
                    fp16_weight_bytes=2 * n * k,
                    fp32_weight_bytes=4 * n * k,
                    scope="3 alternating pairs,3 eager calls; recipe includes validations/host sync; baseline resident dequantized FP32 weight; forward only, no model claim",
                )
            )
            torch.testing.assert_close(fast(), dense(), rtol=5e-5, atol=2e-5)
            passed(f"bounded-paired-forward-timing-{m}-{n}-{k}")
        torch.cuda.synchronize()
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
