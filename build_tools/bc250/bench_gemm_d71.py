#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""D71 GEMM solution bake-off (owned fixtures, private rocBLAS handle).

For the census layouts (mm/addmm, batch 1, up to 40 by count): enumerate
the admitted Tensile solutions via rocblas_gemm_ex_get_solutions, check
every candidate against an independent CPU-double reference (N32 contract:
rtol 0.015, atol 0.05, rel-L2 < 0.004, canary guards, input preservation),
then paired alternating timing (6 rounds x 3 calls) of default vs best
candidate. A torch-level timing on the same tensors exposes any
dispatch-level gap vs the private-handle default.
No global override, no routing change, no new kernels — only existing
admitted solutions, exactly the qualified N32 method.

Additionally a torch-level hipBLASLt AB/BA bake-off on the top shapes:
fresh subprocesses with TORCH_BLAS_PREFER_HIPBLASLT=0/1 (the port routes
fp16 BACKWARD GEMMs to rocBLAS regardless; this measures the forward path).

Usage (guardian-bounded, app environment):
    python3 bench_gemm_d71.py --census <census.json> --output <bench.json>
"""

from __future__ import annotations

import argparse
import ctypes as C
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

V, I = C.c_void_p, C.c_int
DTYPES = {"float16": 150, "float32": 151, "bfloat16": 168}
N, T = 111, 112
CANDIDATE_LIMIT = 32
MAX_LAYOUTS = 40
REL_L2_BOUND = 0.004
RTOL, ATOL = 0.015, 0.05
ROUNDS, CALLS = 6, 3
HIPBLASLT_SHAPES = 3


def status(code):
    if code != 0:
        raise RuntimeError(f"rocBLAS status {code}")


def make_tensor(torch, desc):
    t = torch.empty_strided(desc["shape"], desc["stride"], device="cuda")
    getattr(
        t,
        {"float16": "half", "float32": "float", "bfloat16": "bfloat16"}[desc["dtype"]],
    )()
    t.normal_(0, 0.05)
    return t


class Gemm:
    """N32 private-handle fixture, generalized to FP16/FP32 and beta 0/1."""

    def __init__(self, torch, a, b, bias=None):
        self.torch = torch
        self.a, self.b, self.bias = a, b, bias
        self.m, self.k = a.shape
        _, self.n = b.shape
        self.lib = C.CDLL(
            os.environ.get(
                "BC250_ROCBLAS_LIBRARY", "/workspace/build/dist/rocm/lib/librocblas.so"
            )
        )
        self.lib.rocblas_create_handle.argtypes = [C.POINTER(V)]
        self.lib.rocblas_destroy_handle.argtypes = [V]
        self.lib.rocblas_set_stream.argtypes = [V, V]
        common_types = [
            V,
            I,
            I,
            I,
            I,
            I,
            V,
            V,
            I,
            I,
            V,
            I,
            I,
            V,
            V,
            I,
            I,
            V,
            I,
            I,
            I,
            I,
        ]
        self.lib.rocblas_gemm_ex.argtypes = common_types + [I, C.c_uint32]
        self.lib.rocblas_gemm_ex_get_solutions.argtypes = common_types + [
            C.c_uint32,
            C.POINTER(I),
            C.POINTER(I),
        ]
        self.handle = V()
        status(self.lib.rocblas_create_handle(C.byref(self.handle)))
        status(
            self.lib.rocblas_set_stream(
                self.handle, V(torch.cuda.current_stream().cuda_stream)
            )
        )
        dtype = DTYPES[str(a.dtype).replace("torch.", "")]
        self.storage = torch.full(
            (self.m * self.n + 256,), -77, device="cuda", dtype=a.dtype
        )
        self.out = self.storage[128:-128].view(self.m, self.n)
        self.alpha = C.c_float(1)
        self.beta = C.c_float(1 if bias is not None else 0)
        ta, lda = (N, b.stride(0)) if b.stride(1) == 1 else (T, b.stride(1))
        tb, ldb = (N, a.stride(0)) if a.stride(1) == 1 else (T, a.stride(1))
        assert min(a.stride()) == min(b.stride()) == 1
        c_ptr = V(bias.data_ptr()) if bias is not None else V(self.out.data_ptr())
        ldc = bias.stride(0) if bias is not None else self.n
        self.common = [
            self.handle,
            ta,
            tb,
            self.n,
            self.m,
            self.k,
            C.cast(C.byref(self.alpha), V),
            V(b.data_ptr()),
            dtype,
            lda,
            V(a.data_ptr()),
            dtype,
            ldb,
            C.cast(C.byref(self.beta), V),
            c_ptr,
            dtype,
            ldc,
            V(self.out.data_ptr()),
            dtype,
            self.n,
            DTYPES["float32"],
            1,
        ]
        self.layout = dict(
            transa="N" if ta == N else "T",
            transb="N" if tb == N else "T",
            m=self.n,
            n=self.m,
            k=self.k,
            lda=lda,
            ldb=ldb,
            ldc=ldc,
            beta=int(self.beta.value),
        )

    def solutions(self):
        size = I()
        status(
            self.lib.rocblas_gemm_ex_get_solutions(*self.common, 0, None, C.byref(size))
        )
        assert 0 < size.value < 4096, size.value
        array = (I * size.value)()
        status(
            self.lib.rocblas_gemm_ex_get_solutions(
                *self.common, 0, array, C.byref(size)
            )
        )
        return list(array)[: size.value]

    def call(self, solution):
        status(self.lib.rocblas_gemm_ex(*self.common, solution, 0))
        return self.out

    def guards(self):
        assert bool((self.storage[:128] == -77).all())
        assert bool((self.storage[-128:] == -77).all())

    def close(self):
        status(self.lib.rocblas_destroy_handle(self.handle))


def measure(torch, fn, repeats=CALLS):
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    begin = time.perf_counter()
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return dict(
        gpu_ms=start.elapsed_time(end) / repeats,
        wall_ms=(time.perf_counter() - begin) * 1000 / repeats,
    )


def check_reference(torch, actual, reference):
    actual = actual.cpu().double()
    torch.testing.assert_close(actual, reference, rtol=RTOL, atol=ATOL)
    rel = float((actual - reference).norm() / reference.norm().clamp_min(1e-12))
    if rel >= REL_L2_BOUND:
        raise ValueError(f"rel-L2 {rel} >= {REL_L2_BOUND}")
    return dict(relative_l2=rel, max_abs=float((actual - reference).abs().max()))


def select_layouts(census: dict) -> list[dict]:
    rows = []
    for row in census["layouts"]:
        if row["op"] not in ("mm", "addmm"):
            continue
        rb = row.get("rocblas")
        if rb is None or rb["batch_count"] != 1:
            continue
        if rb["input_type"] not in DTYPES:
            continue
        rows.append(row)
        if len(rows) >= MAX_LAYOUTS:
            break
    return rows


def sweep_layout(torch, row) -> dict:
    descs = row["args"]
    torch.manual_seed(7100 + len(json.dumps(descs)))
    if row["op"] == "addmm":
        # addmm(bias, mat1, mat2)
        bias = make_tensor(torch, descs[0])
        a = make_tensor(torch, descs[1])
        b = make_tensor(torch, descs[2])
    else:
        a = make_tensor(torch, descs[0])
        b = make_tensor(torch, descs[1])
        bias = None
    a0, b0 = a.detach().cpu().clone(), b.detach().cpu().clone()
    bias0 = bias.detach().cpu().clone() if bias is not None else None
    reference = a0.double() @ b0.double()
    if bias is not None:
        reference = reference + bias0.double()
    check_reference(
        torch, (a @ b) if bias is None else torch.addmm(bias, a, b), reference
    )
    native = Gemm(torch, a, b, bias)
    try:
        available = native.solutions()
        tested = []
        for index in [0] + available[:CANDIDATE_LIMIT]:
            native.call(index)
            numerical = check_reference(torch, native.out, reference)
            native.guards()
            timing = measure(torch, lambda: native.call(index))
            tested.append(dict(index=index, **numerical, **timing))
        best = min((r for r in tested if r["index"] != 0), key=lambda r: r["gpu_ms"])
        samples = {"default": [], "candidate": []}
        for round_id in range(ROUNDS):
            order = (
                ["default", "candidate"]
                if round_id % 2 == 0
                else ["candidate", "default"]
            )
            for name in order:
                index = 0 if name == "default" else best["index"]
                samples[name].append(measure(torch, lambda: native.call(index)))
        native.guards()
        assert torch.equal(a.cpu(), a0) and torch.equal(b.cpu(), b0)
        if bias is not None:
            assert torch.equal(bias.cpu(), bias0)
        medians = {
            name: statistics.median(s["gpu_ms"] for s in rows)
            for name, rows in samples.items()
        }
        # Diagnostic: torch-level dispatch on the SAME tensors vs the
        # private-handle default. A large gap means the in-model cost is
        # dispatch-level (flags/path), not solution selection.
        torch_op = (
            (lambda: a @ b) if bias is None else (lambda: torch.addmm(bias, a, b))
        )
        torch_samples = [measure(torch, torch_op) for _ in range(ROUNDS)]
        torch_median = statistics.median(s["gpu_ms"] for s in torch_samples)
        return dict(
            op=row["op"],
            args=descs,
            count=row["count"],
            rocblas_layout=native.layout,
            available_solutions=available,
            tested=tested,
            selected_candidate=best["index"],
            medians=medians,
            torch_mm_ms=torch_median,
            torch_vs_private_default=torch_median / medians["default"],
            paired_gpu_ratio=medians["default"] / medians["candidate"],
        )
    finally:
        native.close()


def hipblaslt_worker(torch, args):
    """One fresh-process torch-level measurement (env set by the parent)."""
    torch.manual_seed(7101)
    desc_a, desc_b = args["a"], args["b"]
    a = make_tensor(torch, desc_a)
    b = make_tensor(torch, desc_b)
    reference = a.detach().cpu().double() @ b.detach().cpu().double()
    check_reference(torch, a @ b, reference)
    for _ in range(3):
        a @ b
    samples = [measure(torch, lambda: a @ b) for _ in range(ROUNDS)]
    result = dict(
        prefer=os.environ.get("TORCH_BLAS_PREFER_HIPBLASLT", "0"),
        median_gpu_ms=statistics.median(s["gpu_ms"] for s in samples),
        samples=samples,
    )
    args_out = Path(args["output"])
    args_out.write_text(json.dumps(result, indent=2) + "\n")


def hipblaslt_bakeoff(torch, layouts, output_dir: Path) -> list[dict]:
    results = []
    for i, row in enumerate(layouts[:HIPBLASLT_SHAPES]):
        if row["op"] != "mm":
            continue
        entry = {"args": row["args"], "count": row["count"], "workers": []}
        for j, prefer in enumerate(("0", "1", "1", "0")):
            out = output_dir / f"hipblaslt-{i}-{j}.json"
            worker_args = {
                "a": row["args"][0],
                "b": row["args"][1],
                "output": str(out),
            }
            env = dict(os.environ)
            env["TORCH_BLAS_PREFER_HIPBLASLT"] = prefer
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--hipblaslt-worker",
                json.dumps(worker_args),
            ]
            with out.with_suffix(".log").open("w") as log:
                subprocess.run(
                    command,
                    env=env,
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=180,
                )
            entry["workers"].append(json.loads(out.read_text()))
        ref = statistics.median(
            w["median_gpu_ms"] for w in entry["workers"] if w["prefer"] == "0"
        )
        cand = statistics.median(
            w["median_gpu_ms"] for w in entry["workers"] if w["prefer"] == "1"
        )
        entry["rocblas_median_ms"] = ref
        entry["hipblaslt_median_ms"] = cand
        entry["hipblaslt_ratio"] = ref / cand
        results.append(entry)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hipblaslt-worker", type=str, default=None)
    args = parser.parse_args()

    import torch

    torch.set_num_threads(4)
    if args.hipblaslt_worker:
        hipblaslt_worker(torch, json.loads(args.hipblaslt_worker))
        return 0
    assert args.census and args.output
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1013"
    census = json.loads(args.census.read_text())
    layouts = select_layouts(census)
    report = dict(
        status="running",
        fixture="bench_gemm_d71",
        bounds={"rtol": RTOL, "atol": ATOL, "rel_l2": REL_L2_BOUND},
        candidate_limit=CANDIDATE_LIMIT,
        note="private handle; existing admitted solutions only; no global change",
        results=[],
        hipblaslt=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for row in layouts:
            report["results"].append(sweep_layout(torch, row))
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            r = report["results"][-1]
            print(
                f"D71_BENCH {r['op']} M{r['rocblas_layout']['m']} "
                f"N{r['rocblas_layout']['n']} K{r['rocblas_layout']['k']} "
                f"default={r['medians']['default']:.3f}ms "
                f"best={r['medians']['candidate']:.3f}ms "
                f"ratio={r['paired_gpu_ratio']:.2f}x "
                f"solution={r['selected_candidate']}",
                flush=True,
            )
        report["hipblaslt"] = hipblaslt_bakeoff(torch, layouts, args.output.parent)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"D71_BENCH_DONE {report['status']}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
