#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare strict W8 implementations with identical numerical/error contracts."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nccl-available", action="store_true")
    parser.add_argument("--pairs", type=int, default=7)
    parser.add_argument("--calls", type=int, default=10)
    args = parser.parse_args()
    assert 3 <= args.pairs <= 31 and 3 <= args.calls <= 100
    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, allowed[:4])
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(541)
    helpers = Path(__file__).resolve().parent
    old = load("quant_b054_old", helpers / "b054-baseline/bc250_quant_b051.py")
    new = load("quant_b054_new", helpers / "b054-candidate/bc250_quant_b051.py")
    report = dict(
        status="running",
        checks=[],
        timings=[],
        errors=[],
        traces=[],
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        pairs=args.pairs,
        calls=args.calls,
    )

    def passed(name):
        report["checks"].append(name)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def exact(actual, expected):
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)

    try:
        assert torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1013:")
        assert torch.distributed.is_nccl_available() == args.nccl_available
        passed("explicit-native-profile")
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for layout in ("contiguous", "strided", "batched"):
                source = torch.randn(13, 17, device="cuda", dtype=dtype)
                po, pn = old.pack(source), new.pack(source)
                exact(po.values, pn.values)
                exact(po.scales, pn.scales)
                xr = torch.randn(3, 17, dtype=dtype) / 4
                bias = torch.randn(13, dtype=dtype) / 8
                up = torch.randn(3, 13, dtype=dtype) / 4
                actuals = []
                for module, p in ((old, po), (new, pn)):
                    if layout == "strided":
                        owner = torch.full((3, 37), 19.0, device="cuda", dtype=dtype)
                        x = owner[:, 1:35:2]
                        x.copy_(xr)
                        sowner = torch.full((29,), 19.0, device="cuda")
                        s = sowner[1:27:2]
                        s.copy_(p.scales)
                        p = module.FrozenInt8Weight(p.values, s)
                    else:
                        x = xr.cuda()
                        if layout == "batched":
                            x = x.reshape(1, 3, 17)
                    x.requires_grad_()
                    b = bias.cuda().requires_grad_()
                    y = module.linear(x, p, b)
                    y.backward(up.cuda().reshape_as(y))
                    actuals.append((y.detach(), x.grad.reshape(3, 17), b.grad))
                for a, b in zip(*actuals):
                    exact(a, b)
                ref = (
                    xr.double()
                    @ (po.values.cpu().double() * po.scales.cpu().double()[:, None]).T
                )
                tol = (
                    dict(rtol=5e-5, atol=3e-6)
                    if dtype == torch.float32
                    else dict(rtol=3e-3, atol=5e-4)
                    if dtype == torch.float16
                    else dict(rtol=2e-2, atol=4e-3)
                )
                torch.testing.assert_close(
                    actuals[1][0].reshape(3, 13).cpu(), ref.to(dtype) + bias, **tol
                )
                passed(
                    "bitwise-pack-output-input-bias-grad-" + str(dtype) + "-" + layout
                )
        # Preserve old error type/message and priority when multiple inputs are bad.
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            x = torch.ones((2, 3), device="cuda", dtype=dtype)
            values = torch.ones((2, 3), device="cuda", dtype=torch.int8)
            for scales in (
                (1.0, 1.0),
                (0.0, 1.0),
                (-1.0, 1.0),
                (float("nan"), 1.0),
                (float("inf"), 1.0),
            ):
                for invalid_input in (None, float("nan"), float("inf"), float("-inf")):
                    scale = torch.tensor(scales, device="cuda")
                    inp = (
                        x
                        if invalid_input is None
                        else torch.full_like(x, invalid_input)
                    )
                    outcomes = []
                    for module in (old, new):
                        try:
                            output = module.linear(
                                inp, module.FrozenInt8Weight(values, scale)
                            )
                        except ValueError as error:
                            outcomes.append((type(error).__name__, str(error)))
                        else:
                            outcomes.append(("ok", output.detach().cpu().tolist()))
                    assert outcomes[0] == outcomes[1], (
                        dtype,
                        scales,
                        invalid_input,
                        outcomes,
                    )
                    if outcomes[0][0] != "ok":
                        report["errors"].append(
                            dict(
                                dtype=str(dtype),
                                scales=str(scales),
                                input=str(invalid_input),
                                outcome=outcomes[0],
                            )
                        )
                    passed(f"error-value-priority-{dtype}-{scales}-{invalid_input}")
            for valid_scales, valid_inputs in (
                (False, False),
                (False, True),
                (True, False),
                (True, True),
            ):
                scale = (
                    torch.ones(2, device="cuda")
                    if valid_scales
                    else torch.zeros(2, device="cuda")
                )
                inp = x if valid_inputs else torch.full_like(x, float("nan"))
                outcomes = []
                for module in (old, new):
                    try:
                        module.linear(
                            inp,
                            module.FrozenInt8Weight(values, scale),
                            torch.ones(3, device="cuda", dtype=dtype),
                        )
                    except ValueError as error:
                        outcomes.append(str(error))
                    else:
                        raise AssertionError("Bad bias accepted")
                assert outcomes[0] == outcomes[1]
                passed(
                    f"scale-input-bias-error-priority-{dtype}-{valid_scales}-{valid_inputs}"
                )
        for m, n, k in ((8, 128, 256), (32, 256, 256), (8, 512, 1024)):
            source = torch.randn(n, k, device="cuda")
            po = old.pack(source)
            pn = new.FrozenInt8Weight(po.values, po.scales)
            x = torch.randn(m, k, device="cuda", requires_grad=True)
            up = torch.randn(m, n, device="cuda")

            def run(module, p, backward):
                if backward:
                    x.grad = None
                result = module.linear(x, p)
                if backward:
                    result.backward(up)
                return result

            for backward in (False, True):
                for module, p in ((old, po), (new, pn)):
                    run(module, p, backward)
                    run(module, p, backward)
                torch.cuda.synchronize()
                pairs = []
                for i in range(args.pairs):
                    sample = {}
                    order = (
                        (("baseline", old, po), ("candidate", new, pn))
                        if i % 2 == 0
                        else (("candidate", new, pn), ("baseline", old, po))
                    )
                    for name, module, p in order:
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        for _ in range(args.calls):
                            run(module, p, backward)
                        torch.cuda.synchronize()
                        sample[name] = (time.perf_counter() - start) * 1000 / args.calls
                    pairs.append(sample)
                bo = run(old, po, backward).detach()
                bg = None if not backward else x.grad.clone()
                cn = run(new, pn, backward).detach()
                exact(bo, cn)
                if backward:
                    exact(bg, x.grad)
                med = {
                    name: statistics.median(p[name] for p in pairs)
                    for name in ("baseline", "candidate")
                }
                report["timings"].append(
                    dict(
                        shape=[m, n, k],
                        backward=backward,
                        pairs_ms=pairs,
                        medians_ms=med,
                        ratio=med["baseline"] / med["candidate"],
                    )
                )
                passed(f"paired-unchanged-result-{m}-{n}-{k}-backward{backward}")
        # Trace after warming each implementation. Same native forward/backward;
        # only strict validation synchronization is expected to differ.
        for name, module, p in (("baseline", old, po), ("candidate", new, pn)):
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                run(module, p, True)
                torch.cuda.synchronize()
            path = args.output.parent / (name + "-trace.json")
            prof.export_chrome_trace(str(path))
            trace = json.loads(path.read_text())
            events = trace["traceEvents"]
            record = dict(
                name=name,
                scalar_reads=sum(
                    e.count
                    for e in prof.key_averages()
                    if e.key == "aten::_local_scalar_dense"
                ),
                stream_syncs=sum(
                    e.get("name") == "hipStreamSynchronize"
                    and e.get("cat") == "cuda_runtime"
                    for e in events
                ),
                kernels=[
                    e["name"]
                    for e in events
                    if e.get("cat") == "kernel"
                    and "weight_int8pack_mm_kernel" in e.get("name", "")
                ],
            )
            assert len(record["kernels"]) == 2, record
            assert record["stream_syncs"] == (2 if name == "baseline" else 1), record
            assert record["scalar_reads"] == (2 if name == "baseline" else 1), record
            report["traces"].append(record)
            passed(name + "-native-kernels-validation-sync-count")
        assert len(report["checks"]) == 90, len(report["checks"])
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
