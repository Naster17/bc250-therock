#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded device QR similarity invariants, failure contracts and phase costs."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import gfx1013_schur_candidate as c

    spec = importlib.util.spec_from_file_location(
        "schur_n26_reference", Path(c.__file__).with_name("schur_n26_reference.py")
    )
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    torch.set_num_threads(4)
    torch.manual_seed(101327)
    report = dict(
        status="running",
        steps=[],
        fixtures=[],
        refusals=[],
        hashes={
            name: hashlib.sha256(
                Path(c.__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in (
                "gfx1013_schur_candidate.py",
                "schur_qr_n27.hip",
                "schur_qr_n27.so",
            )
        },
        method="Three synchronized wall samples/pass, A1/B1/B2/A2. Separate instrumented phase sums include forced synchronization; not additive to uninstrumented timings.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def timed(fn):
        fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(3):
            torch.cuda.synchronize()
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            samples.append(1000 * (time.perf_counter() - start))
        return dict(samples_ms=samples, median_ms=statistics.median(samples))

    def fixture(n, dtype):
        l, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.complex128))
        r, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.complex128))
        x = (l * torch.linspace(1.0, 10.0, n)) @ r.mH
        w = torch.linspace(-2.0, 3.0, n) + 1j * torch.linspace(0.3, 1.7, n)
        return ((x * w) @ torch.linalg.inv(x)).to(dtype)

    def verify(a, result):
        assert result.info == 0
        ev = c.right_eigenvectors(result)
        ref = a.cpu().to(torch.complex128)
        w = ev.values.cpu().to(torch.complex128)
        v = ev.vectors.cpu().to(torch.complex128)
        cw, cv = torch.linalg.eig(ref)
        ix = (w[:, None] - cw[None, :]).abs().argmin(dim=1)
        assert ix.unique().numel() == a.shape[0]
        tol = 8e-4 if a.dtype == torch.complex64 else 6e-10
        torch.testing.assert_close(w, cw[ix], rtol=tol, atol=tol)
        residual = float(
            (
                (ref @ v - v * w).norm(dim=0) / ((ref.norm() + w.abs()) * v.norm(dim=0))
            ).max()
        )
        bound = 8e-6 if a.dtype == torch.complex64 else 4e-12
        assert residual < bound, residual
        direction = max(
            float(
                (
                    v[:, j : j + 1] @ v[:, j : j + 1].mH
                    - cv[:, ix[j : j + 1]] @ cv[:, ix[j : j + 1]].mH
                ).norm()
            )
            for j in range(a.shape[0])
        )
        assert direction < tol, direction
        return dict(
            residual=residual, projector_error=direction, iterations=result.iterations
        )

    # Independent one-step CPU complex128 QR similarity oracle. QR column phases
    # are implementation-dependent, so compare reconstruction and spectral invariants.
    for dtype in (torch.complex64, torch.complex128):
        for n in (2, 4, 9, 16):
            for m in sorted({2, n}):
                for kind in ("general", "singular"):
                    raw = torch.randn(n, n, dtype=dtype)
                    t = torch.triu(raw, diagonal=-1)
                    if m < n:
                        t[m, m - 1] = 0
                    shift = t[m - 1, m - 1].clone()
                    if kind == "singular":
                        t[:m, :m] = torch.eye(m, dtype=dtype) * shift
                    q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.complex128))
                    gt, gq, gs = t.cuda(), q.to(dtype).cuda(), shift.cuda()
                    oldt, oldq = gt.clone(), gq.clone()
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        ot, oq = c._qr_similarity_device(gt, gq, gs, m)
                    torch.cuda.current_stream().wait_stream(stream)
                    ht, hq = ot.cpu().to(torch.complex128), oq.cpu().to(
                        torch.complex128
                    )
                    refq = gq.cpu().to(torch.complex128)
                    reft = gt.cpu().to(torch.complex128)
                    similarity = float(
                        (hq @ ht @ hq.mH - refq @ reft @ refq.mH).norm() / reft.norm()
                    )
                    unitary = float((hq.mH @ hq - torch.eye(n)).norm())
                    fill = float(torch.tril(ht, diagonal=-2).norm() / ht.norm())
                    bound = 128 * n * torch.finfo(t.real.dtype).eps
                    assert max(similarity, unitary, fill) < bound, (
                        similarity,
                        unitary,
                        fill,
                    )
                    # CPU QR control independently obeys the same full similarity.
                    rot, _ = torch.linalg.qr(
                        reft[:m, :m] - complex(shift) * torch.eye(m)
                    )
                    full = torch.eye(n, dtype=torch.complex128)
                    full[:m, :m] = rot
                    cpu_t = full.mH @ reft @ full
                    torch.testing.assert_close(
                        ht.trace(), cpu_t.trace(), rtol=bound, atol=bound
                    )
                    torch.testing.assert_close(
                        torch.linalg.eigvals(ht).abs().sort().values,
                        torch.linalg.eigvals(cpu_t).abs().sort().values,
                        rtol=bound,
                        atol=bound,
                    )
                    torch.testing.assert_close(gt, oldt, rtol=0, atol=0)
                    torch.testing.assert_close(gq, oldq, rtol=0, atol=0)
                    report["steps"].append(
                        dict(
                            dtype=str(dtype),
                            order=n,
                            active=m,
                            kind=kind,
                            similarity=similarity,
                            unitary=unitary,
                            fill=fill,
                            bound=bound,
                            stream="nondefault",
                        )
                    )
        for n in (4, 9, 16):
            a = fixture(n, dtype).cuda()
            before = a.clone()
            legacy = old.schur(a)
            reference = c.schur(a)
            torch.testing.assert_close(reference.t, legacy.t, rtol=0, atol=0)
            torch.testing.assert_close(reference.q, legacy.q, rtol=0, atol=0)
            assert (reference.info, reference.iterations, reference.deflations) == (
                legacy.info,
                legacy.iterations,
                legacy.deflations,
            )
            entry = dict(
                dtype=str(dtype), order=n, correctness={}, paired=[], phases={}
            )
            for backend in ("reference", "device"):
                entry["correctness"][backend] = verify(
                    a, c.schur(a, qr_backend=backend)
                )
            entry["host_control_before"] = timed(lambda: torch.linalg.eig(a))
            for label, backend in (
                ("A1", "reference"),
                ("B1", "device"),
                ("B2", "device"),
                ("A2", "reference"),
            ):
                entry["paired"].append(
                    dict(
                        label=label,
                        backend=backend,
                        **timed(lambda: c.schur(a, qr_backend=backend))
                    )
                )
            # Instrument preparation and iteration separately; leave convergence,
            # scaling/validation and scalar-shift work in a labeled remainder.
            for backend in ("reference", "device"):
                sums = dict(hessenberg_ms=0.0, qr_similarity_ms=0.0, qr_calls=0)
                oldh = c.hessenberg
                name = "_qr_similarity_" + backend
                olds = getattr(c, name)

                def h(*args, **kwargs):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    result = oldh(*args, **kwargs)
                    torch.cuda.synchronize()
                    sums["hessenberg_ms"] += 1000 * (time.perf_counter() - start)
                    return result

                def step(*args, **kwargs):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    result = olds(*args, **kwargs)
                    torch.cuda.synchronize()
                    sums["qr_similarity_ms"] += 1000 * (time.perf_counter() - start)
                    sums["qr_calls"] += 1
                    return result

                c.hessenberg = h
                setattr(c, name, step)
                try:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    result = c.schur(a, qr_backend=backend)
                    torch.cuda.synchronize()
                    sums["instrumented_total_ms"] = 1000 * (time.perf_counter() - start)
                finally:
                    c.hessenberg = oldh
                    setattr(c, name, olds)
                assert sums["qr_calls"] == result.iterations
                sums["other_control_scaling_validation_ms"] = (
                    sums["instrumented_total_ms"]
                    - sums["hessenberg_ms"]
                    - sums["qr_similarity_ms"]
                )
                entry["phases"][backend] = sums
            entry["host_control_after"] = timed(lambda: torch.linalg.eig(a))
            torch.testing.assert_close(a, before, rtol=0, atol=0)
            for budget in (0, 1):
                result = c.schur(a, max_iterations=budget, qr_backend="device")
                assert result.info > 0 and result.iterations == budget
                try:
                    c.right_eigenvectors(result)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("Nonconverged result accepted")
            report["fixtures"].append(entry)
            save()
            print("PASS", str(dtype), n, entry["phases"], flush=True)
    a = torch.eye(2, dtype=torch.complex64, device="cuda")
    try:
        c.schur(a, qr_backend="unknown")
    except ValueError:
        report["refusals"].append("unknown backend")
    else:
        raise AssertionError("Unknown backend accepted")
    # Real HIP activity, not a host solve disguised as a device backend.
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        c._qr_similarity_device(
            a, a, torch.zeros((), dtype=a.dtype, device=a.device), 2
        )
        torch.cuda.synchronize()
    trace = args.output.with_name("trace.json")
    prof.export_chrome_trace(str(trace))
    assert any(
        "small_qr_similarity" in e.get("name", "") and e.get("cat") == "kernel"
        for e in json.loads(trace.read_text())["traceEvents"]
    )
    report["native_kernel_trace"] = str(trace)
    report["status"] = "passed"
    save()


if __name__ == "__main__":
    main()
