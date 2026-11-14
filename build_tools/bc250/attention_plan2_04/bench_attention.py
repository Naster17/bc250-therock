#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PLAN2-04 paired math-vs-tiled attention benchmark (inside the guardian).

Same-input alternated A1/B1/B2/A2 timing (2 warmups + 5 samples, median, HIP
events) for forward, forward+backward and backward-only, plus peak extra
allocation. Parity vs GPU math SDPA is verified before timing each fixture; the
CPU-FP64 anchor lives in qualify_attention.py. Drift > 1.25 marks a fixture
unstable (excluded from speedup claims, raw samples retained). No default or
source change; math stays the production default, tiled stays opt-in.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402
from torch.nn.attention import _gfx1013 as tiled_mod  # noqa: E402

from attention_plan2_04 import fixtures  # noqa: E402

TILED = tiled_mod.scaled_dot_product_attention
REPORT = {"checks": [], "fixtures": [], "environment": {}, "part": None}
STARTED = time.monotonic()
MODES = ("fwd", "fwd_bwd", "bwd")
PROBES = False


def check(name, passed, **detail):
    REPORT["checks"].append({"name": name, "passed": bool(passed), **detail})
    if not passed:
        print(f"ATTN_BENCH_FAIL {name} {detail}", flush=True)
    return passed


def save(output):
    REPORT["seconds"] = time.monotonic() - STARTED
    REPORT["summary"] = {
        "total": len(REPORT["checks"]),
        "passed": sum(c["passed"] for c in REPORT["checks"]),
    }
    REPORT["summary"]["all_passed"] = REPORT["summary"]["passed"] == len(
        REPORT["checks"]
    )
    output.write_text(json.dumps(REPORT, indent=2) + "\n")


def math_sdpa(*xs, **opts):
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(*xs, **opts)


def timed_call(fn, xs, dy, mode, graph_holder, opts):
    if mode == "fwd":
        with torch.no_grad():
            fn(*xs, **opts)
    elif mode == "fwd_bwd":
        y = fn(*xs, **opts)
        torch.autograd.grad(y, xs, dy)
    else:
        torch.autograd.grad(graph_holder[0], xs, dy, retain_graph=True)


def time_pass(fn, xs, dy, mode, label, opts):
    holder = [fn(*xs, **opts)] if mode == "bwd" else [None]
    for _ in range(fixtures.WARMUPS):
        timed_call(fn, xs, dy, mode, holder, opts)
    torch.cuda.synchronize()
    samples = []
    for _ in range(fixtures.SAMPLES):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        timed_call(fn, xs, dy, mode, holder, opts)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    del holder
    return {
        "label": label,
        "median_ms": statistics.median(samples),
        "samples_ms": samples,
    }


def peak_extra(fn, xs, dy, opts):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    y = fn(*xs, **opts)
    torch.autograd.grad(y, xs, dy)
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - baseline
    del y
    return extra


def fixture(row, device, output):
    name, h, hk, d, length, causal, dtype_name = row
    dtype = getattr(torch, dtype_name)
    generator = torch.Generator(device=device).manual_seed(fixtures.SEED)
    xs = [
        torch.randn(
            1, heads, length, d, device=device, dtype=dtype, generator=generator
        )
        * 0.5
        for heads in (h, hk, hk)
    ]
    for x in xs:
        x.requires_grad_()
    dy = torch.randn_like(xs[0], generator=generator)
    opts = dict(is_causal=causal, enable_gqa=h != hk)

    # Parity gate before any timing (tiled vs GPU math, pre-declared bounds).
    tiled_out = TILED(*xs, **opts)
    tiled_grads = torch.autograd.grad(tiled_out, xs, dy)
    math_inputs = [x.detach().clone().requires_grad_() for x in xs]
    math_out = math_sdpa(*math_inputs, **opts)
    math_grads = torch.autograd.grad(math_out, math_inputs, dy)
    rtol, atol = fixtures.CLOSE_BOUNDS[dtype_name]
    close_ok, finite = True, True
    for a, b in zip([tiled_out, *tiled_grads], [math_out, *math_grads]):
        finite &= bool(a.isfinite().all())
        try:
            torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
        except AssertionError:
            close_ok = False
    parity_ok = check(
        f"bench-parity-{name}",
        close_ok and finite,
        close_ok=close_ok,
        finite=finite,
    )
    del tiled_out, tiled_grads, math_out, math_grads, math_inputs
    gc.collect()
    torch.cuda.empty_cache()

    entry = {
        "name": name,
        "dtype": dtype_name,
        "heads": h,
        "kv_heads": hk,
        "head_dim": d,
        "length": length,
        "causal": causal,
        "fwd_mode": "fwd",
        "policy_grad_enabled": fixtures.policy_for(dtype_name, d, True),
        "policy_no_grad": fixtures.policy_for(dtype_name, d, False),
        "modes": {},
        "memory": {},
        "unstable": False,
    }
    if PROBES:
        try:
            math_sdpa(*xs, **opts)
            print(f"ATTN_PROBE {name} pre-timing ok", flush=True)
        except Exception as error:
            print(f"ATTN_PROBE {name} pre-timing FAIL {error!r}", flush=True)
    if parity_ok:
        for mode in MODES:
            passes = []
            for label, fn in (
                ("A1", math_sdpa),
                ("B1", TILED),
                ("B2", TILED),
                ("A2", math_sdpa),
            ):
                passes.append(time_pass(fn, xs, dy, mode, label, opts))
            math_ms = statistics.mean([passes[0]["median_ms"], passes[3]["median_ms"]])
            tiled_ms = statistics.mean([passes[1]["median_ms"], passes[2]["median_ms"]])
            drift_math = passes[3]["median_ms"] / passes[0]["median_ms"]
            drift_tiled = passes[2]["median_ms"] / passes[1]["median_ms"]
            unstable = max(drift_math, drift_tiled) > fixtures.DRIFT_LIMIT
            entry["unstable"] |= unstable
            entry["modes"][mode] = {
                "math_ms": math_ms,
                "tiled_ms": tiled_ms,
                "speedup_math_over_tiled": math_ms / tiled_ms,
                "drift_math": drift_math,
                "drift_tiled": drift_tiled,
                "passes": passes,
            }
            check(
                f"bench-drift-{name}-{mode}",
                not unstable,
                drift_math=drift_math,
                drift_tiled=drift_tiled,
                limit=fixtures.DRIFT_LIMIT,
            )
        for label, fn in (("math", math_sdpa), ("tiled", TILED)):
            entry["memory"][label] = peak_extra(fn, xs, dy, opts)
        entry["memory"]["ratio_tiled_over_math"] = entry["memory"]["tiled"] / max(
            entry["memory"]["math"], 1
        )
    REPORT["fixtures"].append(entry)
    del xs, dy
    gc.collect()
    torch.cuda.empty_cache()
    save(output)
    print(
        f"ATTN_BENCH {name} "
        + " ".join(
            f"{m}={v['speedup_math_over_tiled']:.3f}x"
            for m, v in entry["modes"].items()
        )
        + f" unstable={entry['unstable']}",
        flush=True,
    )


def diagnose(device):
    """Isolate the math-GQA failure trigger (r4/r5 crash diagnostic)."""
    steps = []

    def try_math(tag, tensors, opts, no_grad):
        try:
            if no_grad:
                with torch.no_grad():
                    math_sdpa(*tensors, **opts)
            else:
                math_sdpa(*tensors, **opts)
            steps.append((tag, "ok"))
        except Exception as error:
            steps.append((tag, repr(error)[:160]))

    for causal in (False, True):
        h, hk, d, length = 16, 4, 128, 512
        xs = [
            torch.randn(1, heads, length, d, device=device, dtype=torch.float16) * 0.5
            for heads in (h, hk, hk)
        ]
        for x in xs:
            x.requires_grad_()
        dy = torch.randn_like(xs[0])
        opts = dict(is_causal=causal, enable_gqa=True)
        p = f"causal={causal}"
        try_math(f"{p} 1-fresh-grad", xs, opts, False)
        try_math(f"{p} 2-fresh-nograd", xs, opts, True)
        tiled_out = TILED(*xs, **opts)
        torch.autograd.grad(tiled_out, xs, dy)
        try_math(f"{p} 3-after-tiled-grad", xs, opts, False)
        try_math(f"{p} 4-after-tiled-nograd", xs, opts, True)
        clones = [x.detach().clone().requires_grad_() for x in xs]
        try_math(f"{p} 5-clone-grad", clones, opts, False)
        try_math(f"{p} 6-xs-again-grad", xs, opts, False)
        del tiled_out, clones
        gc.collect()
        torch.cuda.empty_cache()
        try_math(f"{p} 7-xs-after-gc-grad", xs, opts, False)
        try_math(f"{p} 8-xs-after-gc-nograd", xs, opts, True)
        del xs, dy
    print("ATTN_DIAG " + json.dumps(steps), flush=True)


def diagnose2(device, output):
    """Run the real fixtures with a pre-timing math probe on every fixture."""
    global PROBES
    PROBES = True
    for row in fixtures.bench_rows(1):
        fixture(row, device, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--part", type=int, choices=(1, 2), required=True)
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--diagnose2", action="store_true")
    args = parser.parse_args()
    if args.diagnose:
        torch.set_num_threads(4)
        diagnose(torch.device("cuda"))
        args.output.write_text(json.dumps({"status": "diagnose"}) + "\n")
        return 0
    if args.diagnose2:
        torch.set_num_threads(4)
        torch.manual_seed(fixtures.SEED)
        diagnose2(torch.device("cuda"), args.output)
        return 0
    REPORT["part"] = args.part
    REPORT["status"] = "running"
    torch.set_num_threads(4)
    torch.manual_seed(fixtures.SEED)
    device = torch.device("cuda")
    env = REPORT["environment"]
    env["torch"] = torch.__version__
    env["hip"] = torch.version.hip
    env["gcnArchName"] = torch.cuda.get_device_properties(device).gcnArchName
    env["tiled_module_sha256"] = hashlib.sha256(
        Path(tiled_mod.__file__).read_bytes()
    ).hexdigest()
    env["protocol"] = {
        "warmups": fixtures.WARMUPS,
        "samples": fixtures.SAMPLES,
        "order": "A1(math)/B1(tiled)/B2(tiled)/A2(math)",
        "drift_limit": fixtures.DRIFT_LIMIT,
        "fwd_mode": "no_grad for all fixtures; opts (is_causal/enable_gqa) "
        "forwarded to every timed call (r6 fix)",
    }
    check(
        "gfx1013-native",
        env["gcnArchName"].split(":")[0] == "gfx1013",
        arch=env["gcnArchName"],
    )
    probe_q = torch.randn(1, 2, 8, 64, device=device)
    probe_k = torch.randn(1, 1, 8, 64, device=device)
    probe_v = torch.randn_like(probe_k)
    try:
        with torch.no_grad():
            math_sdpa(probe_q, probe_k, probe_v, enable_gqa=True)
        probe_raised, probe_error = False, None
    except RuntimeError as error:
        probe_raised, probe_error = True, str(error)[:200]
    check(
        "math-nograd-gqa-probe-informational",
        True,
        raised=probe_raised,
        error=probe_error,
    )
    del probe_q, probe_k, probe_v
    save(args.output)
    for row in fixtures.bench_rows(args.part):
        fixture(row, device, args.output)
    REPORT["status"] = "passed" if REPORT["summary"]["all_passed"] else "failed"
    save(args.output)
    print(
        f"ATTN_BENCH_DONE part={args.part} "
        f"{REPORT['summary']['passed']}/{REPORT['summary']['total']} "
        f"status={REPORT['status']}",
        flush=True,
    )
    return 0 if REPORT["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
