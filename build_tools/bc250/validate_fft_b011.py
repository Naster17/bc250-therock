#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Explicit BF16 promotion: CPU double/DFT references, gradients and GPU traces."""

import argparse
import itertools
import json
from pathlib import Path

import torch
import bc250_fft as fft


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(1013011)
    assert str(torch.__version__) == "2.13.0a0+gfx1013"
    assert Path(fft.__file__).is_relative_to(
        "/opt/torch-venv/lib/python3.12/site-packages"
    )
    assert fft.__version__ == "0.1.0" and len(fft.__all__) == 18
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1013"
    report = dict(
        status="running",
        checks=[],
        results=[],
        module_file=fft.__file__,
        helper_version=fft.__version__,
        limits=dict(
            output_rtol=5e-4,
            output_atol=5e-4,
            gradient_rtol=0.02,
            gradient_atol=2e-5,
            gradient_relative_l2=0.004,
        ),
    )

    def persist():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def passed(name, **details):
        report["checks"].append(name)
        report["results"].append(dict(name=name, **details))
        persist()

    def loss(value, weights):
        return (value * weights.conj()).real.mean() + 0.03 * value.abs().square().mean()

    def verify(actual, expected, gradient, reference_gradient):
        expected_dtype = torch.complex64 if expected.is_complex() else torch.float32
        assert actual.is_cuda and actual.dtype == expected_dtype
        assert gradient.is_cuda and gradient.dtype == torch.bfloat16
        torch.testing.assert_close(
            actual.cpu().to(expected.dtype), expected, rtol=5e-4, atol=5e-4
        )
        g = gradient.cpu().double()
        torch.testing.assert_close(g, reference_gradient, rtol=0.02, atol=2e-5)
        relative_l2 = float(
            torch.linalg.vector_norm(g - reference_gradient)
            / torch.linalg.vector_norm(reference_gradient).clamp_min(1e-30)
        )
        assert relative_l2 < 0.004, relative_l2
        return dict(
            output_max_absolute=float(
                (actual.cpu().to(expected.dtype) - expected).abs().max()
            ),
            gradient_relative_l2=relative_l2,
        )

    try:
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        for operation, norm, layout, amp in itertools.product(
            fft.__all__,
            ("backward", "forward", "ortho"),
            ("regular", "step"),
            (False, True),
        ):
            raw = (torch.randn(3, 7, 25 if layout == "step" else 11) * 0.2).to(
                torch.bfloat16
            )
            cpu = raw.double().requires_grad_()
            gpu = raw.cuda().detach().requires_grad_()
            before = gpu.detach().clone()
            view = lambda x: x[..., 1:23:2] if layout == "step" else x
            if operation.endswith("2"):
                kw = dict(s=(9, 13), dim=(-2, -1), norm=norm)
            elif operation.endswith("n"):
                kw = dict(s=(5, 13), dim=(0, 2), norm=norm)
            else:
                kw = dict(n=17, dim=-1, norm=norm)
            expected = getattr(torch.fft, operation)(view(cpu), **kw)
            weights = torch.randn(expected.shape, dtype=expected.dtype)
            reference_grad = torch.autograd.grad(loss(expected, weights), cpu)[0]
            stream = streams[int(amp)]
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=amp
            ):
                actual = getattr(fft, operation)(view(gpu), **kw)
                gradient = torch.autograd.grad(
                    loss(actual, weights.to(device="cuda", dtype=actual.dtype)), gpu
                )[0]
            stream.synchronize()
            details = verify(
                actual.detach(), expected.detach(), gradient, reference_grad
            )
            assert torch.equal(gpu.detach(), before)
            if layout == "step":
                assert torch.count_nonzero(gradient[..., ::2]) == 0
                assert torch.count_nonzero(gradient[..., 23:]) == 0
            passed(f"{operation}-{norm}-{layout}-amp{amp}", **details)
        # Non-BF16 values go straight to the original operator.
        x = torch.randn(3, 7, 11, device="cuda")
        for operation in fft.__all__:
            kw = (
                dict(s=(7, 13), dim=(-2, -1))
                if operation.endswith(("2", "n"))
                else dict(n=17)
            )
            expected = getattr(torch.fft, operation)(x, **kw)
            actual = getattr(fft, operation)(x, **kw)
            assert torch.equal(actual, expected)
            passed(operation + "-fp32-pass-through")
        x64 = x.double()
        assert torch.equal(fft.fftn(x64), torch.fft.fftn(x64))
        passed("float64-pass-through")
        complex_input = torch.complex(x, x.flip(-1))
        assert torch.equal(fft.ifftn(complex_input), torch.fft.ifftn(complex_input))
        passed("complex64-pass-through")
        bf = x.to(torch.bfloat16)

        def refuses(native, adapter):
            errors = []
            for function in (native, adapter):
                try:
                    function()
                except (RuntimeError, ValueError, TypeError) as error:
                    errors.append(type(error))
                else:
                    raise AssertionError("Expected native contract refusal")
            assert errors[0] is errors[1]

        refuses(
            lambda: torch.fft.rfft(bf.float(), norm="invalid"),
            lambda: fft.rfft(bf, norm="invalid"),
        )
        passed("native-invalid-norm-contract")
        refuses(
            lambda: torch.fft.fftn(bf.float(), dim=(0, 0)),
            lambda: fft.fftn(bf, dim=(0, 0)),
        )
        passed("native-duplicate-dim-contract")
        requires = bf.detach().requires_grad_()
        target = torch.empty(3, 7, 6, device="cuda", dtype=torch.complex64)
        refuses(
            lambda: torch.fft.rfft(requires.float(), out=target),
            lambda: fft.rfft(requires, out=target),
        )
        passed("native-autograd-out-contract")
        refuses(lambda: torch.fft.fft([1, 2]), lambda: fft.fft([1, 2]))
        passed("native-nontensor-contract")
        for amp in (False, True):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                try:
                    torch.fft.rfft(bf)
                except RuntimeError as error:
                    assert "BFloat16" in str(error)
                else:
                    raise AssertionError("Global torch.fft behavior changed")
        passed("global-bf16-rejection-unchanged-both-amp-contexts")
        # Native out= semantics/guard gaps; no gradient promise for out=.
        for operation, real in (("rfft2", False), ("hfft2", True)):
            value = bf[0]
            expected = getattr(torch.fft, operation)(value.cpu().double(), s=(7, 13))
            dtype = torch.float32 if real else torch.complex64
            sentinel = 0.375 if real else complex(0.375, 0.125)
            backing = torch.full(
                (expected.shape[0], expected.shape[1] * 2 + 3),
                sentinel,
                device="cuda",
                dtype=dtype,
            )
            original = backing.clone()
            out = backing[:, 1 : 1 + 2 * expected.shape[1] : 2]
            actual = getattr(fft, operation)(value, s=(7, 13), out=out)
            assert actual is out
            torch.testing.assert_close(
                out.cpu().to(expected.dtype), expected, rtol=5e-4, atol=5e-4
            )
            mask = torch.ones_like(backing, dtype=torch.bool)
            mask[:, 1 : 1 + 2 * expected.shape[1] : 2] = False
            assert torch.equal(backing[mask], original[mask])
            passed(operation + "-strided-out-guarded")
        # Independent dense DFT, continuous-reference finite differences, then
        # compare the cast-back BF16 leaf gradient (not quantization derivative).
        raw = (
            torch.linspace(-0.2, 0.3, 17) + torch.sin(torch.arange(17) * 0.7) * 0.05
        ).to(torch.bfloat16)
        cpu = raw.double().requires_grad_()
        angle = (
            -2
            * torch.pi
            * torch.arange(17, dtype=torch.float64)[:, None]
            * torch.arange(17, dtype=torch.float64)[None, :]
            / 17
        )
        matrix = torch.complex(angle.cos(), angle.sin())
        expected = matrix @ cpu.cdouble()
        weights = torch.complex(
            torch.linspace(-0.4, 0.8, 17), torch.linspace(0.7, -0.3, 17)
        ).cdouble()
        reference_grad = torch.autograd.grad(loss(expected, weights), cpu)[0]
        finite = []
        for i in (0, 7, 16):
            plus = cpu.detach().clone()
            minus = cpu.detach().clone()
            plus[i] += 1e-5
            minus[i] -= 1e-5
            fd = (
                loss(matrix @ plus.cdouble(), weights)
                - loss(matrix @ minus.cdouble(), weights)
            ) / (2e-5)
            assert abs(float(fd - reference_grad[i])) < 1e-8
            finite.append(float(abs(fd - reference_grad[i])))
        gpu = raw.cuda().requires_grad_()
        actual = fft.fft(gpu)
        gradient = torch.autograd.grad(loss(actual, weights.cfloat().cuda()), gpu)[0]
        details = verify(actual.detach(), expected.detach(), gradient, reference_grad)
        passed(
            "independent-dft17-continuous-reference-finite-differences",
            finite_difference_errors=finite,
            **details,
        )
        # Profile a fresh forward/backward using this adapter and compare again.
        gpu = raw.cuda().requires_grad_()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            actual = fft.fft(gpu)
            gradient = torch.autograd.grad(loss(actual, weights.cfloat().cuda()), gpu)[
                0
            ]
            torch.cuda.synchronize()
        trace = args.output.parent / "fft-b011-trace.json"
        prof.export_chrome_trace(str(trace))
        kernels = [
            e["name"]
            for e in json.loads(trace.read_text())["traceEvents"]
            if e.get("cat") == "kernel"
        ]
        fft_kernels = sorted(
            {
                name
                for name in kernels
                if any(v in name.lower() for v in ("fft", "stockham", "bluestein"))
            }
        )
        assert fft_kernels
        details = verify(actual.detach(), expected.detach(), gradient, reference_grad)
        passed(
            "native-gpu-fft-forward-backward-trace",
            gpu_kernel_count=len(kernels),
            fft_kernel_names=fft_kernels,
            **details,
        )
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        persist()


if __name__ == "__main__":
    main()
