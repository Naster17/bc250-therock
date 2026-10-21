#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded native audio forward/backward checks with independent CPU oracles."""

import argparse
import itertools
import json
from pathlib import Path

import torch
import torchaudio as ta


def recurrence(x, a, b):
    values = []
    for t in range(x.shape[-1]):
        y = sum(b[:, k] * x[..., t - k] for k in range(min(t + 1, b.shape[-1])))
        for k in range(1, min(t + 1, a.shape[-1])):
            y = y - a[:, k] * values[t - k]
        values.append(y / a[:, 0])
    return torch.stack(values, -1)


def dft_power(x):
    padded = torch.nn.functional.pad(x.unsqueeze(0), (8, 8), mode="reflect")[0]
    frames = padded.unfold(-1, 16, 5)
    window = torch.nn.functional.pad(torch.hann_window(12, dtype=x.dtype), (2, 2))
    angle = (
        2
        * torch.pi
        * torch.arange(9, dtype=x.dtype)[:, None]
        * torch.arange(16, dtype=x.dtype)[None, :]
        / 16
    )
    return ((frames * window) @ angle.cos().T).square().transpose(-1, -2) + (
        (frames * window) @ angle.sin().T
    ).square().transpose(-1, -2)


def rnnt_reference(x, labels, times, lengths, fused):
    probs = x.log_softmax(-1) if fused else x
    costs = []
    for n, (time, length) in enumerate(zip(times.tolist(), lengths.tolist())):
        alpha = [[None] * (length + 1) for _ in range(time + 1)]
        alpha[0][0] = probs[n, 0, 0, 0] * 0
        for t in range(time + 1):
            for u in range(length + 1):
                candidates = []
                if t > 0 and alpha[t - 1][u] is not None:
                    candidates.append(alpha[t - 1][u] + probs[n, t - 1, u, 0])
                if u > 0 and t < time and alpha[t][u - 1] is not None:
                    candidates.append(
                        alpha[t][u - 1] + probs[n, t, u - 1, labels[n, u - 1]]
                    )
                if candidates:
                    alpha[t][u] = torch.logsumexp(torch.stack(candidates), 0)
        costs.append(-alpha[time][length])
    return torch.stack(costs)


def collapse(path):
    return [v for i, v in enumerate(path) if v != 0 and (i == 0 or v != path[i - 1])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group", choices=("dsp", "native"), required=True)
    parser.add_argument("--codec-available", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(4042)
    assert str(torch.__version__) == "2.13.0a0+gfx1013"
    assert ta.__version__ == "2.11.0+gfx1013"
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1013"
    report = dict(
        status="running",
        checks=[],
        results=[],
        group=args.group,
        torch=str(torch.__version__),
        torchaudio=ta.__version__,
        source_revision=ta.git_version,
        torchaudio_file=ta.__file__,
    )

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def passed(name, **details):
        torch.cuda.synchronize()
        report["checks"].append(name)
        report["results"].append(dict(name=name, **details))
        save()

    def close(actual, expected, dtype):
        actual = actual.detach().cpu().double()
        expected = expected.detach().cpu().double()
        assert actual.isfinite().all() and expected.isfinite().all()
        rtol, atol, l2 = {
            torch.float64: (2e-8, 2e-10, 1e-7),
            torch.float32: (0.002, 3e-5, 0.003),
            torch.float16: (0.04, 0.01, 0.025),
        }[dtype]
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        relative = float((actual - expected).norm() / expected.norm().clamp_min(1e-12))
        assert relative < l2, relative
        return relative

    def run(
        name, fn, inputs, *, reference=None, strided=False, stream=False, trace=False
    ):
        dtype = inputs[0].dtype
        cpu = [x.detach().double().clone().requires_grad_() for x in inputs]
        expected = (reference or fn)(*cpu)
        up = (
            torch.linspace(-0.7, 0.9, expected.numel())
            .reshape(expected.shape)
            .to(dtype)
        )
        expected.backward(up.double())
        current = torch.cuda.Stream() if stream else torch.cuda.current_stream()
        # Clones are created on their consuming stream, including strided views.
        with torch.cuda.stream(current):
            gpu = []
            for x in inputs:
                if strided and x.ndim >= 2:
                    allocation = torch.empty(
                        (*x.shape[:-1], 2 * x.shape[-1]), dtype=x.dtype, device="cuda"
                    )
                    value = allocation[..., ::2]
                    value.copy_(x)
                else:
                    value = x.cuda()
                gpu.append(value.detach().requires_grad_())
            if trace:
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as prof:
                    actual = fn(*gpu)
                    actual.backward(up.cuda())
                    current.synchronize()
                path = args.output.parent / "native-audio-trace.json"
                prof.export_chrome_trace(str(path))
                kernels = [
                    e
                    for e in json.loads(path.read_text())["traceEvents"]
                    if e.get("cat") == "kernel"
                ]
                native = [e for e in kernels if "iir_" in e["name"].lower()]
                assert len(native) >= 2, [e["name"] for e in kernels]
                streams = {e["args"]["stream"] for e in kernels}
                assert len(streams) == 1, streams
                report["gpu_trace"] = dict(
                    kernel_names=sorted({e["name"] for e in kernels}),
                    native_iir_launches=len(native),
                    streams=sorted(streams),
                )
            else:
                actual = fn(*gpu)
                actual.backward(up.cuda())
            current.synchronize()
        errors = [close(actual, expected, dtype)]
        for a, b in zip(gpu, cpu):
            assert a.grad is not None and b.grad is not None
            errors.append(close(a.grad, b.grad, dtype))
        passed(
            name,
            max_relative_l2=max(errors),
            gradient_tensors=len(gpu),
            strided=strided,
            nondefault_stream=stream,
        )

    try:
        if args.group == "dsp":

            def spectrogram(a):
                return ta.functional.spectrogram(
                    a,
                    pad=0,
                    window=torch.hann_window(12, device=a.device, dtype=a.dtype),
                    n_fft=16,
                    hop_length=5,
                    win_length=12,
                    power=2.0,
                    normalized=False,
                )

            for dtype in (torch.float32, torch.float64):
                x = (torch.randn(2, 96) * 0.1).to(dtype)
                run(
                    "independent-dft-power-gradient-" + str(dtype),
                    spectrogram,
                    [x],
                    reference=dft_power,
                )
                for strided in (False, True):
                    functions = {
                        "spectrogram": spectrogram,
                        "mel": lambda a: ta.transforms.MelSpectrogram(
                            sample_rate=8000, n_fft=32, hop_length=8, n_mels=6
                        ).to(a.device, a.dtype)(a),
                        "mfcc": lambda a: ta.transforms.MFCC(
                            sample_rate=8000,
                            n_mfcc=4,
                            melkwargs=dict(n_fft=32, hop_length=8, n_mels=6),
                        ).to(a.device, a.dtype)(a),
                        "resample-down": lambda a: ta.functional.resample(
                            a, 8000, 6000
                        ),
                        "resample-up": lambda a: ta.functional.resample(a, 8000, 11025),
                        "deltas": lambda a: ta.functional.compute_deltas(
                            a, win_length=5
                        ),
                        "gain": lambda a: ta.functional.gain(a, 3.0),
                        "preemphasis": lambda a: ta.functional.preemphasis(
                            a, coeff=0.83
                        ),
                    }
                    for name, fn in functions.items():
                        run(
                            f"{name}-{dtype}-strided{strided}", fn, [x], strided=strided
                        )
        else:
            assert ta._extension._IS_TORCHAUDIO_EXT_AVAILABLE
            for op in ("_lfilter_core_loop", "rnnt_loss_forward", "forced_align"):
                assert torch._C._dispatch_has_kernel_for_dispatch_key(
                    "torchaudio::" + op, "CUDA"
                ), op
            for dtype in (torch.float32, torch.float64):
                for order in (1, 3):
                    a = torch.tensor(
                        [[1.2, -0.16, 0.03], [0.9, 0.11, -0.02]], dtype=dtype
                    )[:, :order].clone()
                    b = torch.tensor(
                        [[0.4, 0.06, -0.03], [0.3, -0.04, 0.02]], dtype=dtype
                    )[:, :order].clone()
                    x = (torch.randn(2, 2, 37) * 0.1).to(dtype)
                    for strided in (False, True):
                        for stream in (False, True):
                            run(
                                f"iir-{dtype}-order{order}-strided{strided}-stream{stream}",
                                lambda x, a, b: ta.functional.lfilter(
                                    x, a, b, clamp=False
                                ),
                                [x, a, b],
                                reference=recurrence,
                                strided=strided,
                                stream=stream,
                            )
            run(
                "profiled-iir-current-stream-output-all-gradients",
                lambda x, a, b: ta.functional.lfilter(x, a, b, clamp=False),
                [
                    torch.randn(2, 2, 37) * 0.1,
                    torch.tensor([[1.2, -0.16, 0.03], [0.9, 0.11, -0.02]]),
                    torch.tensor([[0.4, 0.06, -0.03], [0.3, -0.04, 0.02]]),
                ],
                reference=recurrence,
                stream=True,
                trace=True,
            )
            for dtype in (torch.float32, torch.float16):
                for fused in (False, True):
                    for time in (4, 35):
                        x = (torch.randn(2, time, 4, 7) * 0.2).to(dtype)
                        if not fused:
                            x = x.float().log_softmax(-1).to(dtype)
                        labels = torch.tensor([[1, 2, 3], [2, 2, 1]], dtype=torch.int32)
                        times = torch.tensor([time, time - 1], dtype=torch.int32)
                        lengths = torch.tensor([3, 2], dtype=torch.int32)
                        run(
                            f"rnnt-{dtype}-fused{fused}-time{time}",
                            lambda a: ta.functional.rnnt_loss(
                                a,
                                labels.to(a.device),
                                times.to(a.device),
                                lengths.to(a.device),
                                blank=0,
                                reduction="none",
                                fused_log_softmax=fused,
                            ),
                            [x],
                            reference=lambda a: rnnt_reference(
                                a, labels, times, lengths, fused
                            ),
                            stream=time == 35,
                        )
            # Validate the dynamic-programming oracle by independently enumerating
            # all tiny monotone RNNT paths and comparing both loss and gradient.
            x = torch.randn(1, 2, 2, 3, dtype=torch.float64, requires_grad=True) * 0.1
            labels = torch.tensor([[1]])
            reference = rnnt_reference(
                x, labels, torch.tensor([2]), torch.tensor([1]), True
            ).sum()
            logp = x.log_softmax(-1)
            paths = [
                logp[0, 0, 0, 1] + logp[0, 0, 1, 0] + logp[0, 1, 1, 0],
                logp[0, 0, 0, 0] + logp[0, 1, 0, 1] + logp[0, 1, 1, 0],
            ]
            independent = -torch.logsumexp(torch.stack(paths), 0)
            torch.testing.assert_close(reference, independent, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(
                torch.autograd.grad(reference, x, retain_graph=True)[0],
                torch.autograd.grad(independent, x)[0],
                atol=1e-12,
                rtol=1e-12,
            )
            passed("rnnt-oracle-versus-enumerated-paths-and-gradients")
            for dtype in (torch.float32, torch.float64, torch.float16):
                for target_dtype in (torch.int32, torch.int64):
                    for time in (7, 105):
                        logp = (torch.randn(1, time, 4) * 0.7).log_softmax(-1).to(dtype)
                        labels = torch.tensor([[1, 1, 2]], dtype=target_dtype)
                        reference = ta.functional.forced_align(logp.double(), labels)
                        with torch.cuda.stream(torch.cuda.Stream()):
                            actual = ta.functional.forced_align(
                                logp.cuda(), labels.cuda()
                            )
                            torch.cuda.current_stream().synchronize()
                        assert torch.equal(actual[0].cpu(), reference[0])
                        close(actual[1], reference[1], dtype)
                        passed(f"forced-align-{dtype}-{target_dtype}-time{time}")
            logp = (torch.randn(1, 7, 3) * 0.7).log_softmax(-1).double()
            target = torch.tensor([[1, 2]])
            candidates = [
                p
                for p in itertools.product(range(3), repeat=7)
                if collapse(p) == [1, 2]
            ]
            scores = [
                sum(float(logp[0, t, v]) for t, v in enumerate(p)) for p in candidates
            ]
            best = candidates[max(range(len(scores)), key=scores.__getitem__)]
            actual = ta.functional.forced_align(logp.cuda(), target.cuda())
            assert actual[0].cpu().tolist() == [list(best)]
            passed("independent-enumerated-ctc-viterbi-path")
            # Explicitly expose the codec dependency boundary, never silently
            # substitute a different implementation of the public file API.
            import importlib.util

            if args.codec_available:
                import torchcodec
                from torchcodec.decoders import AudioDecoder
                from torchcodec.encoders import AudioEncoder

                assert torchcodec.__version__ == "0.17.0+gfx1013"
                assert callable(AudioDecoder) and callable(AudioEncoder)
                passed("native-dispatch-and-explicit-installed-codec-interface")
            else:
                assert importlib.util.find_spec("torchcodec") is None
                for fn, arguments in (
                    (ta.load, ("missing-owned-audio-fixture.wav",)),
                    (
                        ta.save,
                        ("unused-owned-audio-fixture.wav", torch.zeros(1, 16), 8000),
                    ),
                ):
                    try:
                        fn(*arguments)
                    except ImportError as error:
                        assert "torchcodec" in str(error).lower()
                    else:
                        raise AssertionError("Missing codec backend was not reported")
                passed("native-dispatch-and-explicit-unavailable-torchcodec-file-apis")
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
