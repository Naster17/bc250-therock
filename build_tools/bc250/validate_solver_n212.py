#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded N2.12 right-vector recovery profile and independent numerical gates."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--profile-only", action="store_true")
    args = ap.parse_args()
    import torch
    import gfx1013_schur_candidate as c

    torch.set_num_threads(4)
    torch.manual_seed(1013212)
    report = dict(
        status="running",
        fixtures=[],
        direct=[],
        refusals=[],
        source_sha256=hashlib.sha256(Path(c.__file__).read_bytes()).hexdigest(),
        method="Synchronized wall time; warm session; A1/B1/B2/A2 three samples each; creation/close excluded; not kernel-only",
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

    def verify(a, result, backend):
        got = c.right_eigenvectors(result, recovery_backend=backend)
        ref = a.cpu().to(torch.complex128)
        w, v = got.values.cpu().to(torch.complex128), got.vectors.cpu().to(
            torch.complex128
        )
        cw, cv = torch.linalg.eig(ref)
        ix = (w[:, None] - cw[None, :]).abs().argmin(dim=1)
        assert ix.unique().numel() == len(w)
        tol = 8e-4 if a.dtype == torch.complex64 else 6e-10
        torch.testing.assert_close(w, cw[ix], rtol=tol, atol=tol)
        residual = float(
            (
                (ref @ v - v * w).norm(dim=0) / ((ref.norm() + w.abs()) * v.norm(dim=0))
            ).max()
        )
        direction = max(
            float(
                (
                    v[:, j : j + 1] @ v[:, j : j + 1].mH
                    - cv[:, ix[j : j + 1]] @ cv[:, ix[j : j + 1]].mH
                ).norm()
            )
            for j in range(len(w))
        )
        assert residual < (8e-6 if a.dtype == torch.complex64 else 4e-12), residual
        assert direction < tol, direction
        torch.testing.assert_close(
            v.norm(dim=0), torch.ones(len(w), dtype=torch.float64), rtol=tol, atol=tol
        )
        return dict(
            residual=residual,
            projector_error=direction,
            triangular_residual=got.triangular_residual,
        )

    if not args.profile_only:
        spec = importlib.util.spec_from_file_location(
            "schur_n211_reference",
            Path(c.__file__).with_name("schur_n211_reference.py"),
        )
        old = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = old
        spec.loader.exec_module(old)
        report["hashes"] = {
            name: hashlib.sha256(
                Path(c.__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in (
                "gfx1013_schur_candidate.py",
                "gfx1013_gehrd_session.py",
                "schur_backsub_n212.hip",
                "schur_backsub_n212.so",
                "schur_qr_n27.hip",
                "schur_q_n210.so",
                "schur_control_n28.so",
                "schur_chunk_n29.so",
                "schur_n211_reference.py",
            )
        }

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
            entry = dict(
                dtype=str(dtype),
                order=n,
                paired_recovery=[],
                paired_whole=[],
                correctness={},
            )
            with c.GehrdSession() as session:

                def schur():
                    return c.schur(
                        a,
                        qr_backend="device_control",
                        q_backend="device",
                        session=session,
                    )

                result = schur()
                assert result.info == 0
                t0, q0 = result.t.clone(), result.q.clone()
                if args.profile_only:
                    entry["preparation"] = timed(
                        lambda: c.hessenberg(a, q_backend="device", session=session)
                    )
                    entry["schur"] = timed(schur)
                    entry["recovery"] = timed(lambda: c.right_eigenvectors(result))
                    entry["whole"] = timed(lambda: c.right_eigenvectors(schur()))
                    if n == 16:
                        with torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ]
                        ) as prof:
                            c.right_eigenvectors(result)
                        entry["aten_counts"] = {
                            e.key: e.count for e in prof.key_averages()
                        }
                else:
                    legacy = old.right_eigenvectors(
                        old.SchurResult(
                            result.t,
                            result.q,
                            result.info,
                            result.iterations,
                            result.deflations,
                        )
                    )
                    default = c.right_eigenvectors(result)
                    torch.testing.assert_close(
                        default.vectors, legacy.vectors, rtol=0, atol=0
                    )
                    assert (
                        default.relative_separation,
                        default.triangular_residual,
                    ) == (legacy.relative_separation, legacy.triangular_residual)
                    entry["default_exact_n211"] = True
                    for qr in ("reference", "device_control", "device_chunked"):
                        sr = c.schur(
                            a, qr_backend=qr, q_backend="device", session=session
                        )
                        for backend in ("reference", "device"):
                            entry["correctness"][qr + "_" + backend] = verify(
                                a, sr, backend
                            )
                    for label, backend in (
                        ("A1", "reference"),
                        ("B1", "device"),
                        ("B2", "device"),
                        ("A2", "reference"),
                    ):
                        entry["paired_recovery"].append(
                            dict(
                                label=label,
                                backend=backend,
                                **timed(
                                    lambda: c.right_eigenvectors(
                                        result, recovery_backend=backend
                                    )
                                )
                            )
                        )
                        entry["paired_whole"].append(
                            dict(
                                label=label,
                                backend=backend,
                                **timed(
                                    lambda: c.right_eigenvectors(
                                        schur(), recovery_backend=backend
                                    )
                                )
                            )
                        )
                    entry["host_before"] = timed(lambda: torch.linalg.eig(a))
                    # Native kernel alone is diagnostic; public gates remain timed above.
                    unit, _ = c._normal_scale(result.t)
                    scaled = unit / unit.abs().amax()
                    entry["solve_reference"] = timed(
                        lambda: c._backsubstitution_reference(scaled)
                    )
                    entry["solve_device"] = timed(
                        lambda: c._backsubstitution_device(scaled)
                    )
                    if n == 16:
                        entry["aten_counts"] = {}
                        for backend in ("reference", "device"):
                            with torch.profiler.profile(
                                activities=[
                                    torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.CUDA,
                                ]
                            ) as prof:
                                c.right_eigenvectors(result, recovery_backend=backend)
                            entry["aten_counts"][backend] = {
                                e.key: e.count for e in prof.key_averages()
                            }
                    entry["host_after"] = timed(lambda: torch.linalg.eig(a))
                torch.testing.assert_close(result.t, t0, rtol=0, atol=0)
                torch.testing.assert_close(result.q, q0, rtol=0, atol=0)
            torch.testing.assert_close(a, before, rtol=0, atol=0)
            report["fixtures"].append(entry)
            save()
            print("PASS", str(dtype), n, flush=True)
    if args.profile_only:
        report["status"] = "passed"
        save()
        return

    # Direct separated triangular forms cover every admitted order and dtype,
    # strided/conjugate/negative views, input sentinels, and a nondefault stream.
    for dtype in (torch.complex64, torch.complex128):
        for n in range(1, 17):
            host = torch.triu(torch.randn(n, n, dtype=torch.complex128)) * 0.1
            host.diagonal().copy_(
                torch.linspace(-1.0, 1.0, n) + 1j * torch.linspace(0.2, 0.8, n)
            )
            big = torch.full(
                (2 * n + 2, 2 * n + 2), 987 + 321j, dtype=dtype, device="cuda"
            )
            t = big[1 : 2 * n : 2, 1 : 2 * n : 2]
            t.copy_(host.to(dtype).cuda())
            saved = big.clone()
            if n % 3 == 0:
                t = t.conj()
            elif n % 3 == 1:
                t = torch._neg_view(t)
            q = torch.eye(n, dtype=dtype, device="cuda")
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                sr = c.SchurResult(t, q, 0, 0, 0)
                metrics = verify(t, sr, "device")
                got = c.right_eigenvectors(sr, recovery_backend="device")
                ref = c.right_eigenvectors(sr)
                tol = 8e-5 if dtype == torch.complex64 else 8e-13
                torch.testing.assert_close(got.vectors, ref.vectors, rtol=tol, atol=tol)
            stream.synchronize()
            torch.testing.assert_close(big, saved, rtol=0, atol=0)
            report["direct"].append(dict(dtype=str(dtype), order=n, **metrics))
    report["rescaling"] = []
    for dtype in (torch.complex64, torch.complex128):
        n = 16
        step = (
            1.25 * 8 * torch.finfo(torch.empty((), dtype=dtype).real.dtype).eps ** 0.5
        )
        t = torch.triu(torch.ones(n, n, dtype=dtype, device="cuda"), diagonal=1)
        t.diagonal().copy_(torch.arange(n, device="cuda") * step + 0.1j)
        sr = c.SchurResult(t, torch.eye(n, dtype=dtype, device="cuda"), 0, 0, 0)
        metrics = verify(t, sr, "device")
        actual = c.right_eigenvectors(sr, recovery_backend="device")
        expected = c.right_eigenvectors(sr)
        tol = 8e-5 if dtype == torch.complex64 else 8e-13
        torch.testing.assert_close(actual.vectors, expected.vectors, rtol=tol, atol=tol)
        report["rescaling"].append(dict(dtype=str(dtype), order=n, gap=step, **metrics))
    sr = c.SchurResult(
        torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=torch.complex64, device="cuda")),
        torch.eye(3, dtype=torch.complex64, device="cuda"),
        0,
        0,
        0,
    )
    try:
        c.right_eigenvectors(sr, recovery_backend="unknown")
    except ValueError as e:
        assert "backend" in str(e)
        report["refusals"].append("unknown-recovery-backend")
    else:
        raise AssertionError("Unknown backend accepted")
    # Inject private kernel-output faults to prove the unchanged common checks
    # still apply to device results. These are test faults, not real GPU faults.
    original = c._backsubstitution_device
    try:
        for label, fn, message in (
            (
                "nonfinite",
                lambda t: torch.full_like(t, float("nan")),
                "Unstable triangular",
            ),
            ("zero-norm", lambda t: torch.zeros_like(t), "Unstable triangular"),
            ("bad-residual", lambda t: torch.ones_like(t), "residual exceeds"),
        ):
            c._backsubstitution_device = fn
            try:
                c.right_eigenvectors(sr, recovery_backend="device")
            except RuntimeError as e:
                assert message in str(e), str(e)
                report["refusals"].append(label)
            else:
                raise AssertionError("Missing output refusal " + label)
    finally:
        c._backsubstitution_device = original
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        c.right_eigenvectors(sr, recovery_backend="device")
        torch.cuda.synchronize()
    trace = args.output.with_name("trace.json")
    prof.export_chrome_trace(str(trace))
    events = json.loads(trace.read_text())["traceEvents"]
    kernels = [
        e["name"]
        for e in events
        if e.get("cat") == "kernel" and "small_triangular_backsub" in e.get("name", "")
    ]
    assert len(kernels) == 1, kernels
    report["native_kernels"] = kernels
    report["status"] = "passed"
    save()


if __name__ == "__main__":
    main()
