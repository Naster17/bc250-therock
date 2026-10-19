#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded native rocBLAS solution search for captured BF16 model layouts."""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch

V, I = C.c_void_p, C.c_int
BF16, FP32, N, T = 168, 151, 111, 112


def status(code):
    if code != 0:
        raise RuntimeError(f"rocBLAS status {code}")


class Gemm:
    def __init__(self, a, b):
        self.a, self.b = a, b
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
        self.storage = torch.full(
            (self.m * self.n + 256,), -77, device="cuda", dtype=torch.bfloat16
        )
        self.out = self.storage[128:-128].view(self.m, self.n)
        self.alpha, self.beta = C.c_float(1), C.c_float(0)
        ta, lda = (N, b.stride(0)) if b.stride(1) == 1 else (T, b.stride(1))
        tb, ldb = (N, a.stride(0)) if a.stride(1) == 1 else (T, a.stride(1))
        assert min(a.stride()) == min(b.stride()) == 1
        self.common = [
            self.handle,
            ta,
            tb,
            self.n,
            self.m,
            self.k,
            C.cast(C.byref(self.alpha), V),
            V(b.data_ptr()),
            BF16,
            lda,
            V(a.data_ptr()),
            BF16,
            ldb,
            C.cast(C.byref(self.beta), V),
            V(self.out.data_ptr()),
            BF16,
            self.n,
            V(self.out.data_ptr()),
            BF16,
            self.n,
            FP32,
            1,
        ]
        self.layout = dict(
            transa=ta,
            transb=tb,
            m=self.n,
            n=self.m,
            k=self.k,
            lda=lda,
            ldb=ldb,
            ldc=self.n,
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


def measure(fn, repeats=2):
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


def check_reference(actual, reference):
    actual = actual.cpu().double()
    torch.testing.assert_close(actual, reference, rtol=0.015, atol=0.05)
    rel = float((actual - reference).norm() / reference.norm().clamp_min(1e-12))
    assert rel < 0.004, rel
    return dict(relative_l2=rel, max_abs=float((actual - reference).abs().max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-label", default="gemm-n32-layout-r1")
    args = parser.parse_args()
    assert Path(args.audit_label).name == args.audit_label
    torch.set_num_threads(4)
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1013"
    audit = args.output.parent.parent / args.audit_label
    layouts = json.loads((audit / "result.json").read_text())["layouts"]
    result = dict(
        status="running",
        checks=[],
        results=[],
        candidate_limit=32,
        note="Private handle, alpha1/beta0, FP32 accumulation/BF16 output; no global override or routing change",
    )
    try:
        for row in layouts:
            fixture = audit / row["fixture"]
            assert (
                hashlib.sha256(fixture.read_bytes()).hexdigest()
                == row["fixture_sha256"]
            )
            cpu = torch.load(fixture, weights_only=True, map_location="cpu")
            a, b = cpu["a"].cuda(), cpu["b"].cuda()
            assert (
                list(a.stride()) == row["a_stride"]
                and list(b.stride()) == row["b_stride"]
            )
            reference = cpu["a"].double() @ cpu["b"].double()
            check_reference(a @ b, reference)
            native = Gemm(a, b)
            try:
                available = native.solutions()
                tested = []
                # Bound discovery; report omitted solutions rather than imply global tuning.
                for index in [0] + available[:32]:
                    native.call(index)
                    numerical = check_reference(native.out, reference)
                    native.guards()
                    timing = measure(lambda: native.call(index))
                    tested.append(dict(index=index, **numerical, **timing))
                best = min(
                    (r for r in tested if r["index"] != 0), key=lambda r: r["gpu_ms"]
                )
                samples = {"default": [], "candidate": []}
                for round_id in range(6):
                    order = (
                        ["default", "candidate"]
                        if round_id % 2 == 0
                        else ["candidate", "default"]
                    )
                    for name in order:
                        index = 0 if name == "default" else best["index"]
                        samples[name].append(measure(lambda: native.call(index), 3))
                native.guards()
                assert torch.equal(a.cpu(), cpu["a"]) and torch.equal(b.cpu(), cpu["b"])
                medians = {
                    name: statistics.median(s["gpu_ms"] for s in rows)
                    for name, rows in samples.items()
                }
                result["results"].append(
                    dict(
                        fixture=row["fixture"],
                        logical_layout=row,
                        rocblas_layout=native.layout,
                        available_solutions=available,
                        tested=tested,
                        selected_candidate=best["index"],
                        samples=samples,
                        medians=medians,
                        paired_gpu_ratio=medians["default"] / medians["candidate"],
                    )
                )
                result["checks"].append(
                    "cpu-reference-guards-private-handle-paired-" + row["fixture"]
                )
                args.output.write_text(json.dumps(result, indent=2) + "\n")
            finally:
                native.close()
        result["status"] = "passed"
    except Exception as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
