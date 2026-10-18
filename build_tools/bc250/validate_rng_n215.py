#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded GPU RNG state, layouts, offsets, statistics and process continuation."""

import argparse
import itertools
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--group", choices=("state", "statistics", "contracts", "resume"), required=True
    )
    parser.add_argument("--worker", choices=("save", "restore"))
    args = parser.parse_args()
    import torch

    torch.set_num_threads(4)
    arch = torch.cuda.get_device_properties(0).gcnArchName
    assert arch.split(":")[0] == "gfx1013", arch
    seed = 2**63 + 10132150
    report = dict(status="running", arch=arch, checks=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def check(name, execute):
        start = time.monotonic()
        try:
            details = execute()
            torch.cuda.synchronize()
            row = dict(name=name, status="passed", details=details)
        except Exception as error:
            row = dict(name=name, status="failed", error=repr(error))
        row["seconds"] = time.monotonic() - start
        report["checks"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(row, flush=True)

    def equal(actual, expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def sample(out, kind, generator):
        if kind == "uniform":
            out.uniform_(generator=generator)
        elif kind == "normal":
            out.normal_(generator=generator)
        elif kind == "integer":
            out.random_(-3, 17, generator=generator)
        elif kind == "bernoulli":
            out.bernoulli_(0.3, generator=generator)
        elif kind == "exponential":
            out.exponential_(1.7, generator=generator)
        return out

    def state_fixture(kind, dtype, layout):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        generator.set_offset(2**32 + 4)
        saved = generator.get_state().clone()
        assert saved.device.type == "cpu" and saved.dtype == torch.uint8
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        rows, columns, offset = 17, 19, 3
        strides = {
            "contiguous": (columns, 1),
            "step": (columns * 2 + 5, 2),
            "transpose": (1, rows),
        }[layout]
        span = (rows - 1) * strides[0] + (columns - 1) * strides[1] + 1
        outputs, ends = [], []
        for _ in range(2):
            generator.set_state(saved)
            with torch.cuda.stream(stream):
                guard = torch.full(
                    (offset + span + 5,), 123, dtype=dtype, device="cuda"
                )
                out = guard.as_strided((rows, columns), strides, offset)
                sample(out, kind, generator)
                mask = torch.ones(guard.shape, dtype=torch.bool)
                mask.as_strided((rows, columns), strides, offset).fill_(False)
            stream.synchronize()
            assert torch.all(guard.cpu()[mask] == 123)
            outputs.append(out.cpu())
            ends.append(generator.get_state())
        equal(outputs[0], outputs[1])
        equal(ends[0], ends[1])
        assert not torch.equal(saved, ends[0])
        assert torch.isfinite(outputs[0]).all()
        return dict(
            shape=[rows, columns],
            strides=strides,
            exact_replay=True,
            outside_view_unchanged=True,
            offset_after=generator.get_offset(),
        )

    if args.group == "state":
        types = [
            (kind, dtype)
            for kind, dtype in itertools.product(
                ("uniform", "normal", "bernoulli", "exponential"),
                (torch.float16, torch.bfloat16, torch.float32, torch.float64),
            )
        ]
        types += [
            ("integer", torch.int32),
            ("integer", torch.int64),
            ("normal", torch.complex64),
            ("normal", torch.complex128),
        ]
        for (kind, dtype), layout in itertools.product(
            types, ("contiguous", "step", "transpose")
        ):
            check(
                f"{kind}_{dtype}_{layout}", lambda: state_fixture(kind, dtype, layout)
            )

        for dtype, kind, n in itertools.product(
            (torch.float32, torch.float64), ("uniform", "normal"), (0, 1, 3, 257, 65537)
        ):

            def boundary():
                g = torch.Generator(device="cuda").manual_seed(seed)
                state = g.get_state()
                a = sample(torch.empty(n, dtype=dtype, device="cuda"), kind, g)
                after = g.get_state()
                g.set_state(state)
                b = sample(torch.empty_like(a), kind, g)
                equal(a, b)
                equal(g.get_state(), after)
                assert torch.equal(state, after) == (n == 0)
                return dict(elements=n, exact_replay=True, empty_state_unchanged=n == 0)

            check(f"boundary_{kind}_{dtype}_{n}", boundary)

    elif args.group == "statistics":
        for kind, dtype in itertools.product(
            ("uniform", "normal", "bernoulli", "exponential"),
            (torch.float16, torch.bfloat16, torch.float32, torch.float64),
        ):

            def statistics():
                g = torch.Generator(device="cuda").manual_seed(10132150)
                values = (
                    sample(torch.empty(65536, device="cuda", dtype=dtype), kind, g)
                    .double()
                    .cpu()
                )
                assert torch.isfinite(values).all()
                mean, variance = {
                    "uniform": (0.5, 1 / 12),
                    "normal": (0, 1),
                    "bernoulli": (0.3, 0.21),
                    "exponential": (1 / 1.7, 1 / 1.7**2),
                }[kind]
                if kind == "uniform":
                    assert values.min() >= 0 and values.max() < 1
                elif kind == "bernoulli":
                    assert torch.all((values == 0) | (values == 1))
                elif kind == "exponential":
                    assert values.min() >= 0
                # Fixed, conservative moment probes, not a randomness certification.
                observed_mean, observed_var = (
                    values.mean().item(),
                    values.var(unbiased=False).item(),
                )
                mean_bound = 8 * math.sqrt(variance / len(values)) + (
                    0.008 if dtype == torch.bfloat16 else 0.001
                )
                var_bound = 0.04 * max(variance, 0.1)
                assert abs(observed_mean - mean) < mean_bound, (
                    observed_mean,
                    mean,
                    mean_bound,
                )
                assert abs(observed_var - variance) < var_bound, (
                    observed_var,
                    variance,
                    var_bound,
                )
                return dict(
                    mean=observed_mean,
                    variance=observed_var,
                    expected=[mean, variance],
                    bounds=[mean_bound, var_bound],
                    samples=len(values),
                )

            check(f"moments_{kind}_{dtype}", statistics)
        for kind, dtype in itertools.product(
            ("integer", "poisson", "gamma"), (torch.float32, torch.float64)
        ):

            def other_statistics():
                g = torch.Generator(device="cuda").manual_seed(10132151)
                if kind == "integer":
                    values = torch.randint(
                        -3,
                        17,
                        (65536,),
                        device="cuda",
                        generator=g,
                        dtype=torch.int32 if dtype == torch.float32 else torch.int64,
                    ).cpu()
                    assert values.min() >= -3 and values.max() < 17
                    frequencies = torch.bincount(
                        values.long() + 3, minlength=20
                    ).double() / len(values)
                    assert (frequencies - 0.05).abs().max() < 0.004
                    return dict(
                        max_frequency_error=(frequencies - 0.05).abs().max().item()
                    )
                parameters = (0.2, 7.0) if kind == "poisson" else (0.7, 3.5)
                results = []
                for parameter in parameters:
                    tensor = torch.full((65536,), parameter, dtype=dtype, device="cuda")
                    values = (
                        (
                            torch.poisson(tensor, generator=g)
                            if kind == "poisson"
                            else torch._standard_gamma(tensor, generator=g)
                        )
                        .double()
                        .cpu()
                    )
                    assert torch.isfinite(values).all() and values.min() >= 0
                    if kind == "poisson":
                        assert torch.equal(values, values.floor())
                    mean, variance = (
                        values.mean().item(),
                        values.var(unbiased=False).item(),
                    )
                    assert abs(mean - parameter) < 8 * math.sqrt(
                        parameter / len(values)
                    )
                    assert abs(variance - parameter) < 0.06 * parameter
                    results.append(
                        dict(parameter=parameter, mean=mean, variance=variance)
                    )
                return dict(moments=results)

            check(f"moments_{kind}_{dtype}", other_statistics)

    elif args.group == "contracts":
        for kind, offset in itertools.product(
            ("uniform", "normal", "integer"), (0, 4, 256, 2**32 + 12, 2**40 + 4)
        ):

            def offset_fixture():
                g = torch.Generator(device="cuda").manual_seed(seed)
                g.set_offset(offset)
                dtype = torch.int64 if kind == "integer" else torch.float64
                a = sample(torch.empty(257, device="cuda", dtype=dtype), kind, g)
                after = g.get_state()
                advance = g.get_offset() - offset
                assert advance > 0 and advance % 4 == 0
                g2 = torch.Generator(device="cuda").manual_seed(seed)
                g2.set_offset(offset)
                b = sample(torch.empty_like(a), kind, g2)
                equal(a, b)
                equal(g2.get_state(), after)
                g2.manual_seed(seed)
                assert g2.get_offset() == 0
                return dict(offset=offset, advance=advance, exact_replay=True)

            check(f"offset_{kind}_{offset}", offset_fixture)

        def refusals():
            g = torch.Generator(device="cuda").manual_seed(seed)
            before = g.get_state()
            rejected = 0
            for offset in (1, 3, 6):
                try:
                    g.set_offset(offset)
                except RuntimeError as error:
                    assert "multiple of 4" in str(error)
                    rejected += 1
                else:
                    raise AssertionError("Unaligned offset accepted")
                equal(g.get_state(), before)
            for state in (
                torch.zeros(7, dtype=torch.uint8),
                torch.zeros(15, dtype=torch.uint8),
                torch.zeros(16, dtype=torch.float32),
            ):
                try:
                    g.set_state(state)
                except (RuntimeError, TypeError):
                    rejected += 1
                else:
                    raise AssertionError("Malformed state accepted")
                g.set_state(before)
            g.set_offset(256)
            g.set_state(before[:8].clone())
            assert g.get_offset() == 0 and g.initial_seed() == seed
            return dict(refusals=rejected, legacy_seed_only_state=True)

        check("offset_and_state_contracts", refusals)

        def stream_sequence():
            streams = [torch.cuda.Stream(), torch.cuda.Stream()]
            for stream in streams:
                stream.wait_stream(torch.cuda.current_stream())

            def run(private):
                g = torch.Generator(device="cuda").manual_seed(seed)
                outputs = []
                for index, n in enumerate((3, 257, 65537, 17)):
                    stream = (
                        streams[index % 2] if private else torch.cuda.current_stream()
                    )
                    with torch.cuda.stream(stream):
                        outputs.append(torch.randn(n, device="cuda", generator=g))
                torch.cuda.synchronize()
                return [v.cpu() for v in outputs], g.get_state()

            ordinary, state = run(False)
            private, other = run(True)
            for a, b in zip(ordinary, private):
                equal(a, b)
            equal(state, other)
            return dict(submissions=4, private_streams=2, host_order_preserved=True)

        check("shared_generator_ordered_submissions_two_streams", stream_sequence)

        def isolation():
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(10132152)
                cpu, gpu = torch.get_rng_state(), torch.cuda.get_rng_state()
                g = torch.Generator(device="cuda").manual_seed(seed)
                torch.rand(257, device="cuda", generator=g)
                equal(torch.cuda.get_rng_state(), gpu)
                for fail in (False, True):
                    try:
                        with torch.random.fork_rng(devices=[0]):
                            torch.rand(257, device="cuda")
                            torch.randn(17)
                            if fail:
                                raise ValueError("intentional fork_rng test")
                    except ValueError:
                        assert fail
                    equal(torch.get_rng_state(), cpu)
                    equal(torch.cuda.get_rng_state(), gpu)
            return dict(
                explicit_generator_isolated=True, fork_normal_and_exception_restore=True
            )

        check("default_and_explicit_generator_isolation", isolation)

        def wide_integer():
            g = torch.Generator(device="cuda").manual_seed(seed)
            a = torch.randint(0, 2**40, (65537,), generator=g, device="cuda")
            first_offset = g.get_offset()
            b = torch.randint(0, 2**40, (65537,), generator=g, device="cuda")
            assert g.get_offset() > first_offset
            values = torch.cat((a, b)).cpu()
            unique = values.unique().numel()
            assert values.min() >= 0 and values.max() < 2**40
            assert unique > 131000, unique
            assert (values >> 32).unique().numel() > 240
            return dict(
                samples=values.numel(),
                unique=unique,
                range_bits=40,
                distinct_first_second=not torch.equal(a, b),
            )

        check("wide_integer_consecutive_calls_no_bulk_state_reuse", wide_integer)

        def trace_dropout():
            torch.manual_seed(10132153)
            x = torch.linspace(-1, 1, 1025, device="cuda").requires_grad_()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            saved = torch.cuda.get_rng_state()
            outputs = []
            for replay in range(2):
                torch.cuda.set_rng_state(saved)
                activities = [
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
                with torch.profiler.profile(activities=activities) as prof:
                    with torch.cuda.stream(stream):
                        out, mask = torch.ops.aten.native_dropout(x, 0.25, True)
                        grad = torch.autograd.grad(out.sum(), x)[0]
                        values = torch.randn(257, device="cuda")
                    stream.synchronize()
                expected_mask = mask.cpu().to(x.dtype) / 0.75
                torch.testing.assert_close(
                    out.cpu(), x.detach().cpu() * expected_mask, rtol=2e-6, atol=2e-6
                )
                torch.testing.assert_close(
                    grad.cpu(), expected_mask, rtol=2e-6, atol=2e-6
                )
                outputs.append(
                    (
                        out.detach().cpu(),
                        mask.cpu(),
                        grad.cpu(),
                        values.cpu(),
                        torch.cuda.get_rng_state(),
                    )
                )
                if replay == 0:
                    trace = args.output.parent / "rng-trace.json"
                    prof.export_chrome_trace(str(trace))
                    kernels = [
                        e
                        for e in json.loads(trace.read_text())["traceEvents"]
                        if e.get("cat") == "kernel"
                    ]
                    names = sorted({e["name"] for e in kernels})
                    assert any(
                        any(
                            s in name.lower()
                            for s in ("distribution", "normal", "dropout")
                        )
                        for name in names
                    ), names
            for a, b in zip(*outputs):
                equal(a, b)
            return dict(
                kernels=len(kernels),
                kernel_names=names,
                dropout_output_gradient_reference=True,
            )

        check("dropout_output_gradient_replay_gpu_trace", trace_dropout)

    else:

        def continuation(g):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                result = dict(
                    explicit_uniform=torch.rand(257, device="cuda", generator=g),
                    explicit_normal=torch.randn(17, device="cuda", generator=g),
                    explicit_integer=torch.randint(
                        0, 2**40, (259,), device="cuda", generator=g
                    ),
                    default_uniform=torch.rand(263, device="cuda"),
                    default_normal=torch.randn(3, device="cuda"),
                )
            stream.synchronize()
            result = {k: v.cpu() for k, v in result.items()}
            result.update(
                cpu=torch.randn(17),
                python=torch.tensor(
                    [random.random() for _ in range(5)], dtype=torch.float64
                ),
                explicit_state=g.get_state(),
                default_state=torch.cuda.get_rng_state(),
                cpu_state=torch.get_rng_state(),
            )
            return result

        checkpoint = args.output.parent / "rng-checkpoint.pt"
        expected_path = args.output.parent / "rng-expected.pt"
        if args.worker == "save":
            torch.manual_seed(10132154)
            random.seed(10132154)
            g = torch.Generator(device="cuda").manual_seed(seed)
            g.set_offset(2**40 + 4)
            torch.randn(3, generator=g, device="cuda")
            torch.randn(3, device="cuda")
            torch.randn(3)
            random.random()
            torch.save(
                dict(
                    explicit=g.get_state(),
                    default=torch.cuda.get_rng_state(),
                    cpu=torch.get_rng_state(),
                    python=random.getstate(),
                ),
                checkpoint,
            )
            torch.save(continuation(g), expected_path)
            return
        if args.worker == "restore":
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            g = torch.Generator(device="cuda")
            g.set_state(state["explicit"])
            torch.cuda.set_rng_state(state["default"])
            torch.set_rng_state(state["cpu"])
            random.setstate(state["python"])
            actual = continuation(g)
            expected = torch.load(expected_path, map_location="cpu", weights_only=True)
            for name in expected:
                equal(actual[name], expected[name])
            args.output.write_text(
                json.dumps(dict(status="passed", exact_fields=list(expected))) + "\n"
            )
            return

        def resume():
            for worker in ("save", "restore"):
                command = [
                    sys.executable,
                    __file__,
                    "--group",
                    "resume",
                    "--worker",
                    worker,
                    "--output",
                    str(args.output.parent / (worker + ".json")),
                ]
                with (args.output.parent / (worker + ".log")).open("w") as log:
                    subprocess.run(
                        command,
                        check=True,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=100,
                    )
            return json.loads((args.output.parent / "restore.json").read_text())

        check("fresh_process_mixed_rng_checkpoint_continuation", resume)

    report["status"] = (
        "passed" if all(r["status"] == "passed" for r in report["checks"]) else "failed"
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    assert report["status"] == "passed"


if __name__ == "__main__":
    main()
