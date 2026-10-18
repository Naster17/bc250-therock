#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""N2.8 one-step control oracle, CPU eigenpair references and paired wall cost."""

import argparse
from collections import Counter
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
    args = ap.parse_args()
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    import gfx1013_schur_candidate as c

    torch.set_num_threads(4)
    torch.manual_seed(101328)
    spec = importlib.util.spec_from_file_location(
        "schur_n27_reference", Path(c.__file__).with_name("schur_n27_reference.py")
    )
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
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
                "schur_control_n28.hip",
                "schur_control_n28.so",
                "schur_qr_n27.so",
                "schur_n27_reference.py",
            )
        },
        method="Three synchronized wall samples per A1/B1/B2/A2 pass. ATen counts separate from timing; status transfers are explicitly counted.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def oracle(t, q, state, budget):
        t, q = t.clone(), q.clone()
        m, it, defs, since, phase = state
        finfo = torch.finfo(t.real.dtype)
        while m > 1:
            threshold = (
                finfo.eps * (t[m - 2, m - 2].abs() + t[m - 1, m - 1].abs()) + finfo.tiny
            )
            if not bool(t[m - 1, m - 2].abs() <= threshold):
                break
            t[m - 1, m - 2] = 0
            m -= 1
            defs += 1
            since = 0
        if m == 1:
            return t, q, [m, it, defs, since, 1]
        if it == budget:
            return t, q, [m, it, defs, since, 2]
        aa, bb, cc, dd = (
            t[m - 2, m - 2],
            t[m - 2, m - 1],
            t[m - 1, m - 2],
            t[m - 1, m - 1],
        )
        middle = (aa + dd) / 2
        root = torch.sqrt(((aa - dd) / 2).square() + bb * cc)
        s1, s2 = middle + root, middle - root
        shift = s1 if bool((s1 - dd).abs() <= (s2 - dd).abs()) else s2
        if since and since % 20 == 0:
            shift = dd + cc.abs()
        t, q = c._qr_similarity_device(t, q, shift, m)
        fill = torch.tril(t, diagonal=-2).norm()
        if not bool(torch.isfinite(t).all()) or not bool(
            fill <= 128 * finfo.eps * t.norm()
        ):
            return t, q, [m, it, defs, since, 3]
        return torch.triu(t, diagonal=-1), q, [m, it + 1, defs, since + 1, 0]

    def step_case(t, q, state, budget, label, expected_phase=None):
        before_t, before_q = t.clone(), q.clone()
        ds = torch.tensor(state, dtype=torch.int32, device=t.device)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            ot, oq = c._control_step_device(t, q, ds, budget)
        torch.cuda.current_stream().wait_stream(stream)
        actual = ds.cpu().tolist()
        if expected_phase is not None:
            assert actual[4] == expected_phase, (label, actual)
            error = None
        else:
            rt, rq, expected = oracle(t, q, state, budget)
            assert actual == expected, (label, actual, expected)
            tol = 128 * t.shape[0] * torch.finfo(t.real.dtype).eps
            # QR column phases are not unique, especially for a rank-deficient
            # shifted active block. Align Q, then apply the same diagonal
            # similarity to T; retain the exact original numerical bound.
            phases = (rq.conj() * oq).sum(dim=0)
            torch.testing.assert_close(
                phases.abs(), torch.ones_like(phases.real), rtol=tol, atol=tol
            )
            phases = phases / phases.abs()
            aligned_t = phases.conj()[:, None] * rt * phases[None, :]
            torch.testing.assert_close(ot, aligned_t, rtol=tol, atol=tol)
            torch.testing.assert_close(oq, rq * phases[None, :], rtol=tol, atol=tol)
            error = float((ot - aligned_t).norm())
            if actual[4] == 0:
                reconstruction = oq @ ot @ oq.mH
                original = q @ t @ q.mH
                relative = float((reconstruction - original).norm() / original.norm())
                assert relative < tol, (label, relative, tol)
        torch.testing.assert_close(t, before_t, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(q, before_q, rtol=0, atol=0)
        report["steps"].append(
            dict(
                label=label,
                dtype=str(t.dtype),
                state=state,
                budget=budget,
                result_state=actual,
                oracle_t_error=error,
                stream="nondefault",
            )
        )
        save()

    for dtype in (torch.complex64, torch.complex128):
        for n in (2, 4, 9, 16):
            t = torch.triu(torch.randn(n, n, dtype=dtype, device="cuda"), diagonal=-1)
            q = torch.eye(n, dtype=dtype, device="cuda")
            for m in sorted({2, n}):
                for since in (0, 20, 40):
                    active_t = t.clone()
                    if m < n:
                        active_t[m, m - 1] = 0
                    step_case(
                        active_t,
                        q,
                        [m, 0, n - m, since, 0],
                        5,
                        f"n{n}_m{m}_since{since}",
                    )
            step_case(t, q, [n, 0, 0, 0, 0], 0, f"n{n}_budget0")
            step_case(
                torch.diag(t.diag()),
                q,
                [n, 0, 0, 19, 0],
                0,
                f"n{n}_deflate_before_budget",
            )
        eps = torch.finfo(t.real.dtype).eps
        q = torch.eye(2, dtype=dtype, device="cuda")
        for factor in (0.5, 1.0, 2.0):
            t = torch.tensor(
                [[1, 0.3], [2 * eps * factor, 1]], dtype=dtype, device="cuda"
            )
            step_case(t, q, [2, 0, 0, 0, 0], 1, f"deflation_boundary_{factor}")
        # Branch/tie controls for principal complex square root and exact rank loss.
        for label, raw in [
            ("rotation", [[0, -1], [1, 0]]),
            ("zero_root", [[1, 0], [1, 1]]),
            ("negative_zero", [[0, 1], [-1, 0]]),
        ]:
            step_case(
                torch.tensor(raw, dtype=dtype, device="cuda"),
                q,
                [2, 0, 0, 0, 0],
                1,
                label,
            )
        t = torch.tensor([[float("nan"), 1], [1, 0]], dtype=dtype, device="cuda")
        step_case(t, q, [2, 0, 0, 0, 0], 1, "nonfinite_guard", 3)
        t = torch.ones(4, 4, dtype=dtype, device="cuda")
        q = torch.eye(4, dtype=dtype, device="cuda")
        step_case(t, q, [4, 0, 0, 0, 0], 1, "fill_guard", 3)
    save()

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

    def verify(a, values, vectors):
        ref = a.cpu().to(torch.complex128)
        w = values.cpu().to(torch.complex128)
        v = vectors.cpu().to(torch.complex128)
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
        return dict(residual=residual, projector_error=direction)

    class Counts(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.counts = Counter()

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.counts[str(func)] += 1
            return func(*args, **(kwargs or {}))

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
            ref = c.schur(a)
            legacy = old.schur(a)
            torch.testing.assert_close(ref.t, legacy.t, rtol=0, atol=0)
            torch.testing.assert_close(ref.q, legacy.q, rtol=0, atol=0)
            assert (ref.info, ref.iterations, ref.deflations) == (
                legacy.info,
                legacy.iterations,
                legacy.deflations,
            )
            entry = dict(
                dtype=str(dtype),
                order=n,
                reference_exact_n27=True,
                correctness={},
                paired_schur=[],
                paired_whole=[],
                profile={},
            )
            for backend in ("reference", "device", "device_control"):
                schur = c.schur(a, qr_backend=backend)
                assert schur.info == 0
                ev = c.right_eigenvectors(schur)
                entry["correctness"][backend] = dict(
                    iterations=schur.iterations,
                    deflations=schur.deflations,
                    **verify(a, ev.values, ev.vectors),
                )
                with Counts() as counts:
                    c.schur(a, qr_backend=backend)
                torch.cuda.synchronize()
                entry["profile"][backend] = dict(
                    scalar_reads=counts.counts.get(
                        "aten._local_scalar_dense.default", 0
                    ),
                    aten_calls=sum(counts.counts.values()),
                    operators=dict(counts.counts),
                )
            entry["correctness"]["host"] = verify(a, *torch.linalg.eig(a))
            entry["host_before"] = timed(lambda: torch.linalg.eig(a))
            for label, backend in (
                ("A1", "device"),
                ("B1", "device_control"),
                ("B2", "device_control"),
                ("A2", "device"),
            ):
                entry["paired_schur"].append(
                    dict(
                        label=label,
                        backend=backend,
                        **timed(lambda: c.schur(a, qr_backend=backend)),
                    )
                )
                entry["paired_whole"].append(
                    dict(
                        label=label,
                        backend=backend,
                        **timed(
                            lambda: c.right_eigenvectors(c.schur(a, qr_backend=backend))
                        ),
                    )
                )
            entry["host_after"] = timed(lambda: torch.linalg.eig(a))
            for budget in (0, 1):
                result = c.schur(a, max_iterations=budget, qr_backend="device_control")
                assert result.info > 0 and result.iterations == budget
                try:
                    c.right_eigenvectors(result)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("Partial result accepted")
            torch.testing.assert_close(a, before, rtol=0, atol=0)
            report["fixtures"].append(entry)
            save()
            print("PASS", str(dtype), n, entry["correctness"], flush=True)
    a = torch.eye(2, dtype=torch.complex64, device="cuda")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        result = c.schur(
            torch.tensor([[1.0, 2.0], [0.3, 4.0]], dtype=a.dtype, device=a.device),
            qr_backend="device_control",
        )
        assert result.info == 0
        torch.cuda.synchronize()
    trace = args.output.with_name("trace.json")
    prof.export_chrome_trace(str(trace))
    events = json.loads(trace.read_text())["traceEvents"]
    assert any(
        e.get("cat") == "kernel" and "small_schur_control" in e.get("name", "")
        for e in events
    )
    report["native_kernel_trace"] = str(trace)
    report["status"] = "passed"
    save()


if __name__ == "__main__":
    main()
