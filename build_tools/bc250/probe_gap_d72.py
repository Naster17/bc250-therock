#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""D72 GEMM in-model-vs-fixture gap diagnostic (owned fixtures, measurement only).

D71 located the LoRA step's dominant cost in the large base projections
(~35 ms/call in-model vs ~8.8 ms hot-cache fixture, 4x) but never explained it.
D71's private-handle sweep always passed flags=0 (the FORWARD path); the port
sets rocblas_gemm_flags_fp16_alt_impl (0x4) on BACKWARD fp16 GEMMs
(aten/src/ATen/cuda/CUDABlas.cpp), which the fixture never exercised. This tool
tests the three discriminating hypotheses with isolated microbenchmarks (no
full-model profile, avoiding the D71 r6 host-thrash vector):

  alt-impl : flags=0 vs flags=0x4 solution enumeration + parity + paired timing
             on the top census mm layouts (the untested backward path).
  sustained: the dominant GEMM in a ~36 s tight loop, per-window ms/call plus
             best-effort hwmon clock/temp (thermal/clock throttling test). The
             throughput-over-time signal is primary; clock reads corroborate.
  locality : the dominant GEMM with hot reused vs cold pooled vs freshly
             written inputs (memory-locality test).

Reuses the qualified N32/D71 private-handle Gemm (bench_gemm_d71) and the N32
parity contract (rtol 0.015 / atol 0.05 / rel-L2 < 0.004). No global override,
no routing/source/default/release change, no new kernels, no user data.

Usage (guardian-bounded):
    python3 probe_gap_d72.py --mode alt-impl  --census c.json --output o.json
    python3 probe_gap_d72.py --mode sustained --census c.json --output o.json
    python3 probe_gap_d72.py --mode locality  --census c.json --output o.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import bench_gemm_d71 as bench

# rocblas_gemm_flags_fp16_alt_impl (rocblas-types.h). The port applies it to
# backward fp16 GEMMs via ROCmBackwardPassGuard (CUDABlas.cpp); D71 used 0.
FP16_ALT_IMPL = 0x4
MEANINGFUL_RATIO = 1.5
DOMINANT_RATIO = 2.5
SUSTAINED_SECONDS = 36.0
SUSTAINED_WINDOW = 2.0
LOCALITY_POOL_PAIRS = 8
MAX_LAYOUTS = 8
ROUNDS = bench.ROUNDS


class GapGemm(bench.Gemm):
    """N32/D71 private-handle Gemm with an explicit rocBLAS flags argument."""

    def __init__(self, torch, a, b, bias=None, flags=0):
        super().__init__(torch, a, b, bias)
        self.flags = flags

    def solutions(self):
        size = bench.I()
        bench.status(
            self.lib.rocblas_gemm_ex_get_solutions(
                *self.common, self.flags, None, bench.C.byref(size)
            )
        )
        assert 0 < size.value < 4096, size.value
        array = (bench.I * size.value)()
        bench.status(
            self.lib.rocblas_gemm_ex_get_solutions(
                *self.common, self.flags, array, bench.C.byref(size)
            )
        )
        return list(array)[: size.value]

    def call(self, solution):
        bench.status(self.lib.rocblas_gemm_ex(*self.common, solution, self.flags))
        return self.out


def select_mm_layouts(census: dict, limit: int = MAX_LAYOUTS) -> list[dict]:
    rows = [
        row
        for row in census["layouts"]
        if row["op"] == "mm"
        and row.get("rocblas", {}).get("batch_count") == 1
        and row["rocblas"]["input_type"] == "float16"
    ]
    rows.sort(key=lambda r: r["count"], reverse=True)
    return rows[:limit]


def dominant_profile_op(profile: dict) -> dict:
    """The D71-measured dominant mm by in-model device time (decision-grade).

    Geometry proxies tie (2048x6144 == 6144x2048), so the dominant is taken
    from D71's per-shape in-model profile, not from the census geometry.
    """
    mm_ops = [
        op
        for op in profile["gemm_ops"]
        if op["op"] == "aten::mm" and len(op["input_shapes"]) == 2
    ]
    return max(mm_ops, key=lambda op: op["self_device_ms"])


def dominant_layout(census: dict, profile: dict) -> dict:
    top = dominant_profile_op(profile)
    shapes = [list(s) for s in top["input_shapes"]]
    for row in select_mm_layouts(census, limit=64):
        if [d["shape"] for d in row["args"]] == shapes:
            return row
    raise ValueError(f"dominant profile shape {shapes} not found in census")


def make_pair(torch, row):
    descs = row["args"]
    torch.manual_seed(7200 + len(json.dumps(descs)))
    a = bench.make_tensor(torch, descs[0])
    b = bench.make_tensor(torch, descs[1])
    return a, b


def paired_default_timing(torch, fwd: GapGemm, bwd: GapGemm) -> dict:
    samples = {"forward": [], "backward": []}
    for round_id in range(ROUNDS):
        order = (
            ["forward", "backward"] if round_id % 2 == 0 else ["backward", "forward"]
        )
        for name in order:
            gemm = fwd if name == "forward" else bwd
            samples[name].append(bench.measure(torch, lambda: gemm.call(0)))
    return {
        name: statistics.median(s["gpu_ms"] for s in rows)
        for name, rows in samples.items()
    }


def sweep_flags(torch, row) -> dict:
    a, b = make_pair(torch, row)
    reference = a.detach().cpu().double() @ b.detach().cpu().double()
    bench.check_reference(torch, a @ b, reference)
    per_flag = {}
    for name, flags in (("forward", 0), ("backward", FP16_ALT_IMPL)):
        gemm = GapGemm(torch, a, b, flags=flags)
        try:
            available = gemm.solutions()
            tested = []
            for index in [0] + available[: bench.CANDIDATE_LIMIT]:
                gemm.call(index)
                numerical = bench.check_reference(torch, gemm.out, reference)
                gemm.guards()
                timing = bench.measure(torch, lambda: gemm.call(index))
                tested.append(dict(index=index, **numerical, **timing))
            best = min(
                (r for r in tested if r["index"] != 0), key=lambda r: r["gpu_ms"]
            )
            per_flag[name] = dict(
                flags=flags,
                available_solutions=available,
                tested=tested,
                default_ms=tested[0]["gpu_ms"],
                best_index=best["index"],
                best_ms=best["gpu_ms"],
            )
        finally:
            gemm.close()
    fwd = GapGemm(torch, a, b, flags=0)
    bwd = GapGemm(torch, a, b, flags=FP16_ALT_IMPL)
    try:
        medians = paired_default_timing(torch, fwd, bwd)
        fwd.guards()
        bwd.guards()
    finally:
        fwd.close()
        bwd.close()
    return dict(
        args=row["args"],
        count=row["count"],
        rocblas_layout=row["rocblas"],
        forward=per_flag["forward"],
        backward=per_flag["backward"],
        paired_default_ms=medians,
        backward_forward_ratio=medians["backward"] / medians["forward"],
        best_backward_forward_ratio=per_flag["backward"]["best_ms"]
        / per_flag["forward"]["best_ms"],
        solution_sets_differ=(
            set(per_flag["forward"]["available_solutions"])
            != set(per_flag["backward"]["available_solutions"])
        ),
    )


def read_clock_temp() -> dict:
    """Best-effort read-only hwmon sensors (N4-validated paths). Never writes."""
    try:
        hw = {
            (p / "name").read_text().strip(): p
            for p in Path("/sys/class/hwmon").glob("hwmon*")
        }
        amdgpu = hw.get("amdgpu")
        if amdgpu is None:
            return dict(freq_hz=None, temp_c=None)
        return dict(
            freq_hz=int((amdgpu / "freq1_input").read_text()),
            temp_c=int((amdgpu / "temp1_input").read_text()) / 1000.0,
        )
    except Exception:
        return dict(freq_hz=None, temp_c=None)


def window_call_count(ms_est: float, window_seconds: float = SUSTAINED_WINDOW) -> int:
    """Bounded call count for one synchronized ~window_seconds GPU-time window."""
    return max(50, min(2000, int(window_seconds * 1000.0 / max(ms_est, 0.01))))


def sustained(torch, row, seconds: float = SUSTAINED_SECONDS) -> dict:
    a, b = make_pair(torch, row)
    gemm = GapGemm(torch, a, b, flags=0)
    windows = []
    try:
        for _ in range(5):
            gemm.call(0)
        torch.cuda.synchronize()
        # Calibrate a fixed call count per ~SUSTAINED_WINDOW-second window so
        # each window is a bounded, synchronized batch. (An earlier wall-clock
        # inner loop queued an unbounded async backlog whose single sync ate the
        # whole budget, yielding one window — not a throttle measurement.)
        calibration = bench.measure(torch, lambda: gemm.call(0), repeats=10)
        calls_per_window = window_call_count(calibration["gpu_ms"])
        begin = time.perf_counter()
        while time.perf_counter() - begin < seconds:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(calls_per_window):
                gemm.call(0)
            end.record()
            end.synchronize()
            elapsed_ms = start.elapsed_time(end)
            windows.append(
                dict(
                    t=round(time.perf_counter() - begin, 3),
                    calls=calls_per_window,
                    ms_per_call=elapsed_ms / calls_per_window,
                    **read_clock_temp(),
                )
            )
        gemm.guards()
    finally:
        gemm.close()
    head = statistics.median(w["ms_per_call"] for w in windows[:3])
    tail = statistics.median(w["ms_per_call"] for w in windows[-3:])
    freqs = [w["freq_hz"] for w in windows if w["freq_hz"] is not None]
    temps = [w["temp_c"] for w in windows if w["temp_c"] is not None]
    return dict(
        args=row["args"],
        seconds=seconds,
        windows=windows,
        head_ms=head,
        tail_ms=tail,
        degradation_ratio=tail / head,
        min_ms=min(w["ms_per_call"] for w in windows),
        max_ms=max(w["ms_per_call"] for w in windows),
        freq_hz_min=min(freqs) if freqs else None,
        freq_hz_max=max(freqs) if freqs else None,
        temp_c_max=max(temps) if temps else None,
        clock_visible=bool(freqs),
    )


def locality(torch, row, pool_pairs: int = LOCALITY_POOL_PAIRS) -> dict:
    a, b = make_pair(torch, row)
    reference = a.detach().cpu().double() @ b.detach().cpu().double()
    bench.check_reference(torch, a @ b, reference)
    hot = GapGemm(torch, a, b, flags=0)
    pool_a = [a.clone() for _ in range(pool_pairs)]
    pool_b = [b.clone() for _ in range(pool_pairs)]
    cold = [GapGemm(torch, pa, pb, flags=0) for pa, pb in zip(pool_a, pool_b)]
    fresh_src = a.clone()
    fresh = GapGemm(torch, a, b, flags=0)
    try:
        for gemm in [hot, *cold, fresh]:
            gemm.call(0)
        torch.cuda.synchronize()
        samples = {"hot": [], "cold": [], "fresh": []}
        cursor = 0
        for round_id in range(ROUNDS):
            for name in ("hot", "cold", "fresh"):
                if name == "hot":
                    fn = lambda: hot.call(0)
                elif name == "cold":
                    gemm = cold[cursor % pool_pairs]
                    cursor += 1
                    fn = lambda: gemm.call(0)
                else:

                    def fn():
                        a.copy_(fresh_src)
                        fresh.call(0)

                samples[name].append(bench.measure(torch, fn))
        hot.guards()
        fresh.guards()
    finally:
        for gemm in [hot, *cold, fresh]:
            gemm.close()
    medians = {
        name: statistics.median(s["gpu_ms"] for s in rows)
        for name, rows in samples.items()
    }
    return dict(
        args=row["args"],
        pool_pairs=pool_pairs,
        medians=medians,
        cold_hot_ratio=medians["cold"] / medians["hot"],
        fresh_hot_ratio=medians["fresh"] / medians["hot"],
    )


def run_mode(torch, mode: str, census: dict, profile: dict) -> dict:
    dom = dominant_layout(census, profile)
    in_model_ms = dominant_profile_op(profile)["device_ms_per_call"]
    if mode == "alt-impl":
        others = [r for r in select_mm_layouts(census) if r["args"] != dom["args"]]
        layouts = [dom] + others[: MAX_LAYOUTS - 1]
        results = [sweep_flags(torch, row) for row in layouts]
        return dict(
            mode=mode,
            results=results,
            dominant=results[0],
            dominant_in_model_ms_per_call=in_model_ms,
        )
    if mode == "sustained":
        return dict(
            mode=mode,
            dominant_in_model_ms_per_call=in_model_ms,
            **sustained(torch, dom),
        )
    if mode == "locality":
        return dict(
            mode=mode, dominant_in_model_ms_per_call=in_model_ms, **locality(torch, dom)
        )
    raise ValueError(f"unknown mode {mode}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", required=True, choices=["alt-impl", "sustained", "locality"]
    )
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch

    torch.set_num_threads(4)
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1013"
    census = json.loads(args.census.read_text())
    profile = json.loads(args.profile.read_text())
    report = dict(
        status="running",
        fixture="probe_gap_d72",
        mode=args.mode,
        bounds={
            "rtol": bench.RTOL,
            "atol": bench.ATOL,
            "rel_l2": bench.REL_L2_BOUND,
            "meaningful_ratio": MEANINGFUL_RATIO,
            "dominant_ratio": DOMINANT_RATIO,
        },
        note="measurement only; private handle; no global/source/release change",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report.update(run_mode(torch, args.mode, census, profile))
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.mode == "alt-impl":
        dom = report["dominant"]
        print(
            f"D72_ALTIMPL dominant bf_ratio={dom['backward_forward_ratio']:.3f}x "
            f"fwd={dom['paired_default_ms']['forward']:.3f}ms "
            f"bwd={dom['paired_default_ms']['backward']:.3f}ms "
            f"solutions_differ={dom['solution_sets_differ']}",
            flush=True,
        )
    elif args.mode == "sustained":
        print(
            f"D72_SUSTAINED head={report['head_ms']:.3f}ms tail={report['tail_ms']:.3f}ms "
            f"degradation={report['degradation_ratio']:.3f}x "
            f"clock_visible={report['clock_visible']} temp_max={report['temp_c_max']}",
            flush=True,
        )
    else:
        print(
            f"D72_LOCALITY hot={report['medians']['hot']:.3f}ms "
            f"cold={report['medians']['cold']:.3f}ms fresh={report['medians']['fresh']:.3f}ms "
            f"cold_hot={report['cold_hot_ratio']:.3f}x fresh_hot={report['fresh_hot_ratio']:.3f}x",
            flush=True,
        )
    print(f"D72_{args.mode.upper()}_DONE {report['status']}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
