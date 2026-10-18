#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""N2.10 GEHRD/Q phase profile and bounded device reflector-Q references."""

import argparse
import hashlib
import json
import importlib.util
import statistics
import sys
from pathlib import Path
import time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--profile-only", action="store_true")
    args = ap.parse_args()
    import torch
    import gfx1013_schur_candidate as c

    torch.set_num_threads(4)
    torch.manual_seed(1013210)
    report = dict(
        status="running",
        preparation=[],
        source_sha256=hashlib.sha256(Path(c.__file__).read_bytes()).hexdigest(),
        method="Separate synchronized GEHRD and reflector-Q wall phases; three diagnostic samples, not kernel-only timing",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    for dtype in (torch.complex64, torch.complex128):
        for n in (4, 9, 16):
            raw = torch.randn(n, n, dtype=torch.complex128).to(dtype)
            a = raw.cuda()
            c.hessenberg(a)
            torch.cuda.synchronize()
            samples = []
            for _ in range(3):
                phases = {}
                torch.cuda.synchronize()
                start = time.perf_counter()
                h, q = c.hessenberg(a, phase_times=phases)
                torch.cuda.synchronize()
                phases["total_ms"] = 1000 * (time.perf_counter() - start)
                samples.append(phases)
            ref = a.cpu().to(torch.complex128)
            hh = h.cpu().to(torch.complex128)
            qq = q.cpu().to(torch.complex128)
            residual = float((ref - qq @ hh @ qq.mH).norm() / ref.norm())
            unitary = float((qq.mH @ qq - torch.eye(n)).norm())
            bound = 128 * n * torch.finfo(raw.real.dtype).eps
            assert max(residual, unitary) < bound, (residual, unitary, bound)
            report["preparation"].append(
                dict(
                    dtype=str(dtype),
                    order=n,
                    samples=samples,
                    residual=residual,
                    unitary=unitary,
                    bound=bound,
                )
            )
            save()
            print("PREPARATION", str(dtype), n, samples, flush=True)
    if args.profile_only:
        report["status"] = "passed"
        save()
        return

    spec = importlib.util.spec_from_file_location(
        "schur_n29_reference", Path(c.__file__).with_name("schur_n29_reference.py")
    )
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    report["hashes"] = {
        name: hashlib.sha256(Path(c.__file__).with_name(name).read_bytes()).hexdigest()
        for name in (
            "gfx1013_schur_candidate.py",
            "schur_q_n210.hip",
            "schur_q_n210.so",
            "schur_qr_n27.hip",
            "schur_control_n28.so",
            "schur_chunk_n29.so",
            "schur_n29_reference.py",
        )
    }
    report["reflectors"], report["fixtures"], report["refusals"] = [], [], []
    for dtype in (torch.complex64, torch.complex128):
        for n in (2, 4, 9, 16):
            for kind in ("complex", "zero", "mixed"):
                packed = torch.randn(n, n, dtype=torch.complex128)
                taus = []
                for i in range(n - 1):
                    v = torch.zeros(n, dtype=torch.complex128)
                    v[i + 1] = 1
                    v[i + 2 :] = 0.2 * torch.randn(n - i - 2, dtype=torch.complex128)
                    packed[i + 2 :, i] = v[i + 2 :]
                    theta = 0.3 + 0.1 * i
                    tau = (
                        1 - torch.exp(torch.tensor(theta * 1j, dtype=torch.complex128))
                    ) / (v.abs().square().sum())
                    taus.append(
                        0j
                        if kind == "zero" or (kind == "mixed" and i % 2 == 0)
                        else complex(tau)
                    )
                packed = packed.to(dtype)
                tau = torch.tensor(taus, dtype=dtype)
                ref = torch.eye(n, dtype=torch.complex128)
                for i in range(n - 1):
                    v = torch.zeros(n, dtype=torch.complex128)
                    v[i + 1] = 1
                    v[i + 2 :] = packed[i + 2 :, i].to(torch.complex128)
                    ref = ref @ (
                        torch.eye(n, dtype=torch.complex128)
                        - tau[i].to(torch.complex128) * v[:, None] * v.conj()[None, :]
                    )
                backing = torch.full(
                    (n + 2, 2 * n + 3), complex(311, -197), dtype=dtype, device="cuda"
                )
                view = backing[1 : n + 1, 1 : 2 * n + 1 : 2]
                view.copy_(packed)
                tau_backing = torch.full(
                    (2 * n + 1,), complex(311, -197), dtype=dtype, device="cuda"
                )
                tau_view = tau_backing[1 : 2 * n - 1 : 2]
                tau_view.copy_(tau)
                before, before_tau = backing.clone(), tau_backing.clone()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    q = c._form_q_device(view, tau_view)
                torch.cuda.current_stream().wait_stream(stream)
                actual = q.cpu().to(torch.complex128)
                bound = 32 * n * torch.finfo(packed.real.dtype).eps
                torch.testing.assert_close(actual, ref, rtol=bound, atol=bound)
                unitary = float((actual.mH @ actual - torch.eye(n)).norm())
                assert unitary < 128 * n * torch.finfo(packed.real.dtype).eps
                torch.testing.assert_close(backing, before, rtol=0, atol=0)
                torch.testing.assert_close(tau_backing, before_tau, rtol=0, atol=0)
                report["reflectors"].append(
                    dict(
                        dtype=str(dtype),
                        order=n,
                        kind=kind,
                        cpu_error=float((actual - ref).norm()),
                        unitary=unitary,
                        bound=bound,
                        stream="nondefault",
                        input_guards_unchanged=True,
                    )
                )
    save()

    def verify(a, result):
        assert result.info == 0
        ev = c.right_eigenvectors(result)
        ref = a.cpu().to(torch.complex128)
        w, v = ev.values.cpu().to(torch.complex128), ev.vectors.cpu().to(
            torch.complex128
        )
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
        assert residual < (8e-6 if a.dtype == torch.complex64 else 4e-12), residual
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

    for dtype in (torch.complex64, torch.complex128):
        for n in (4, 9, 16):
            l, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.complex128))
            r, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.complex128))
            x = (l * torch.linspace(1.0, 10.0, n, dtype=torch.float64)) @ r.mH
            w = torch.linspace(-2.0, 3.0, n, dtype=torch.float64) + 1j * torch.linspace(
                0.3, 1.7, n, dtype=torch.float64
            )
            a = ((x * w) @ torch.linalg.inv(x)).to(dtype).cuda()
            before = a.clone()
            reference = c.schur(a)
            legacy = old.schur(a)
            torch.testing.assert_close(reference.t, legacy.t, rtol=0, atol=0)
            torch.testing.assert_close(reference.q, legacy.q, rtol=0, atol=0)
            assert (reference.info, reference.iterations, reference.deflations) == (
                legacy.info,
                legacy.iterations,
                legacy.deflations,
            )
            entry = dict(
                dtype=str(dtype),
                order=n,
                reference_exact_n29=True,
                correctness={},
                preparation_profile={},
                paired_preparation=[],
                paired_whole=[],
            )
            for qr in ("reference", "device_control", "device_chunked"):
                for qb in ("reference", "device"):
                    entry["correctness"][qr + "_q_" + qb] = verify(
                        a, c.schur(a, qr_backend=qr, q_backend=qb)
                    )
            hr, qr = c.hessenberg(a)
            hd, qd = c.hessenberg(a, q_backend="device")
            torch.testing.assert_close(hd, hr, rtol=0, atol=0)
            bound = 128 * n * torch.finfo(a.real.dtype).eps
            torch.testing.assert_close(qd, qr, rtol=bound, atol=bound)
            for qb in ("reference", "device"):
                phases = {}
                c.hessenberg(a, q_backend=qb, phase_times=phases)
                entry["preparation_profile"][qb] = phases
            entry["host_before"] = timed(lambda: torch.linalg.eig(a))
            for label, qb in (
                ("A1", "reference"),
                ("B1", "device"),
                ("B2", "device"),
                ("A2", "reference"),
            ):
                entry["paired_preparation"].append(
                    dict(
                        label=label,
                        q_backend=qb,
                        **timed(lambda: c.hessenberg(a, q_backend=qb))
                    )
                )
                entry["paired_whole"].append(
                    dict(
                        label=label,
                        q_backend=qb,
                        **timed(
                            lambda: c.right_eigenvectors(
                                c.schur(a, qr_backend="device_control", q_backend=qb)
                            )
                        )
                    )
                )
            entry["host_after"] = timed(lambda: torch.linalg.eig(a))
            torch.testing.assert_close(a, before, rtol=0, atol=0)
            report["fixtures"].append(entry)
            save()
            print("PASS", str(dtype), n, entry["preparation_profile"], flush=True)
    for dtype in (torch.complex64, torch.complex128):
        for n in (1, 4):
            a = torch.zeros(n, n, dtype=dtype, device="cuda")
            hr, qr = c.hessenberg(a)
            hd, qd = c.hessenberg(a, q_backend="device")
            torch.testing.assert_close(hd, hr, rtol=0, atol=0)
            torch.testing.assert_close(qd, qr, rtol=0, atol=0)
    a = torch.eye(2, dtype=torch.complex64, device="cuda")
    for fn in (c.schur, c.hessenberg):
        try:
            fn(a, q_backend="unknown")
        except ValueError:
            report["refusals"].append(fn.__name__)
        else:
            raise AssertionError("Unknown Q backend accepted")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        c.hessenberg(a, q_backend="device")
        torch.cuda.synchronize()
    trace = args.output.with_name("trace.json")
    prof.export_chrome_trace(str(trace))
    assert any(
        e.get("cat") == "kernel" and "small_gehrd_form_q" in e.get("name", "")
        for e in json.loads(trace.read_text())["traceEvents"]
    )
    report["native_kernel_trace"] = str(trace)
    report["status"] = "passed"
    save()


if __name__ == "__main__":
    main()
