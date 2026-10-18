#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""N2.11 binding/handle/allocation phase profile and bounded device reflector-Q references."""

import argparse
import hashlib
import json
import importlib.util
import statistics
import sys
import threading
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
    torch.manual_seed(1013211)
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
            c.hessenberg(a, q_backend="device")
            torch.cuda.synchronize()
            samples = []
            for _ in range(3):
                phases = {}
                torch.cuda.synchronize()
                start = time.perf_counter()
                h, q = c.hessenberg(a, q_backend="device", phase_times=phases)
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
        "schur_n210_reference", Path(c.__file__).with_name("schur_n210_reference.py")
    )
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    report["hashes"] = {
        name: hashlib.sha256(Path(c.__file__).with_name(name).read_bytes()).hexdigest()
        for name in (
            "gfx1013_schur_candidate.py",
            "gfx1013_gehrd_session.py",
            "schur_q_n210.so",
            "schur_control_n28.so",
            "schur_chunk_n29.so",
            "schur_n210_reference.py",
        )
    }
    report["lifetime"], report["fixtures"], report["refusals"] = [], [], []
    a = torch.randn(9, 9, dtype=torch.complex64, device="cuda")

    def refusal(label, fn, text):
        try:
            fn()
        except (RuntimeError, ValueError) as e:
            assert text in str(e), (label, str(e))
            report["refusals"].append(label)
        else:
            raise AssertionError("Missing refusal: " + label)

    sess = c.GehrdSession()
    with sess:
        h, q = c.hessenberg(a, q_backend="device", session=sess)
        original = sess.handle.value
        for dtype in (torch.complex64, torch.complex128) * 3:
            ht, qt = c.hessenberg(a.to(dtype), q_backend="device", session=sess)
            hf, qf = c.hessenberg(a.to(dtype), q_backend="device")
            torch.testing.assert_close(ht, hf, rtol=0, atol=0)
            torch.testing.assert_close(qt, qf, rtol=0, atol=0)
            assert sess.handle.value == original
        refusal("nested-context", lambda: sess.__enter__(), "cannot be nested")
        sess._busy = True
        refusal("busy-use", lambda: c.hessenberg(a, session=sess), "already in use")
        refusal("busy-close", sess.close, "active GEHRD")
        sess._busy = False
        refusal(
            "device-mismatch",
            lambda: sess._check_context(torch.device("cuda", 1)),
            "device mismatch",
        )
        saved = sess._pid
        sess._pid = -1
        refusal("pid-guard-injected", lambda: c.schur(a, session=sess), "after fork")
        sess._pid = saved
        errors = []

        def other_thread():
            try:
                c.hessenberg(a, session=sess)
            except RuntimeError as e:
                errors.append(str(e))

        thread = threading.Thread(target=other_thread)
        thread.start()
        thread.join(timeout=5)
        assert (
            not thread.is_alive()
            and len(errors) == 1
            and "creating host thread" in errors[0]
        )
        report["refusals"].append("other-host-thread")
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            refusal(
                "wrong-stream",
                lambda: c.hessenberg(a, session=sess),
                "creating GPU stream",
            )
        # Rejection must not poison the valid owning context.
        c.hessenberg(a, q_backend="device", session=sess)
        completed = torch.cuda.Event()
        completed.record(sess.stream)
        destroy = sess.bindings.destroy
        destroy_calls = []

        def checked_destroy(handle):
            assert completed.query(), "Stream must complete before native destruction"
            destroy_calls.append(handle.value)
            return destroy(handle)

        sess.bindings.destroy = checked_destroy
    assert sess.closed and sess.handle.value is None and destroy_calls == [original]
    sess.close()
    assert destroy_calls == [original]
    for fn in (c.hessenberg, c.schur):
        for raw in (a, a[:1, :1], torch.zeros_like(a)):
            refusal(
                "closed-" + fn.__name__ + str(tuple(raw.shape)),
                lambda fn=fn, raw=raw: fn(raw, session=sess),
                "closed",
            )
    report["lifetime"].append(
        dict(
            name="repeated-mixed-dtype-handle-reuse/close-sync/idempotence",
            uses=sess.uses,
            passed=True,
        )
    )
    # Failure after native GEHRD submission must release the busy flag, then
    # synchronize and destroy in the context manager even while propagating.
    fail = c.GehrdSession()
    old_form = c._form_q_device

    def fail_q(*args):
        raise RuntimeError("injected after GEHRD")

    try:
        with fail:
            c._form_q_device = fail_q
            c.hessenberg(a, q_backend="device", session=fail)
    except RuntimeError as e:
        assert "injected after GEHRD" in str(e)
    else:
        raise AssertionError("Injected failure did not propagate")
    finally:
        c._form_q_device = old_form
    assert fail.closed and fail.handle.value is None and not fail._busy
    report["lifetime"].append(
        dict(name="exception-after-native-submission-cleanup", passed=True)
    )
    # Two caller-owned sessions on two streams, interleaved without explicit
    # device synchronization between submissions. Handles/workspaces stay private.
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    sessions = []
    outputs = []
    for st in streams:
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            sessions.append(c.GehrdSession())
    try:
        assert sessions[0].handle.value != sessions[1].handle.value
        for i in range(6):
            k = i % 2
            with torch.cuda.stream(streams[k]):
                outputs.append(
                    (k, c.hessenberg(a, q_backend="device", session=sessions[k]))
                )
        for se in sessions:
            se.close()
        for k, (hh, qq) in outputs:
            hf, qf = c.hessenberg(a, q_backend="device")
            torch.testing.assert_close(hh, hf, rtol=0, atol=0)
            torch.testing.assert_close(qq, qf, rtol=0, atol=0)
    finally:
        for se in sessions:
            se.close()
    report["lifetime"].append(
        dict(
            name="two-private-stream-sessions/interleaved-six-submissions", passed=True
        )
    )
    refusal(
        "invalid-session", lambda: c.hessenberg(a, session=object()), "Expected a GEHRD"
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
                reference_exact_n210=True,
                correctness={},
                preparation_profile={},
                paired_preparation=[],
                paired_whole=[],
            )
            started = time.perf_counter()
            session = c.GehrdSession()
            entry["session_create_ms"] = 1000 * (time.perf_counter() - started)
            with session:
                for qr in ("reference", "device_control", "device_chunked"):
                    for reused in (False, True):
                        result = c.schur(
                            a,
                            qr_backend=qr,
                            q_backend="device",
                            session=session if reused else None,
                        )
                        entry["correctness"][qr + "_session_" + str(reused)] = verify(
                            a, result
                        )
                hr, qr = c.hessenberg(a, q_backend="device")
                hs, qs = c.hessenberg(a, q_backend="device", session=session)
                torch.testing.assert_close(hs, hr, rtol=0, atol=0)
                torch.testing.assert_close(qs, qr, rtol=0, atol=0)
                for reused in (False, True):
                    phases = {}
                    c.hessenberg(
                        a,
                        q_backend="device",
                        session=session if reused else None,
                        phase_times=phases,
                    )
                    entry["preparation_profile"][
                        "session" if reused else "fresh"
                    ] = phases
                entry["host_before"] = timed(lambda: torch.linalg.eig(a))
                for label, reused in (
                    ("A1", False),
                    ("B1", True),
                    ("B2", True),
                    ("A2", False),
                ):
                    entry["paired_preparation"].append(
                        dict(
                            label=label,
                            session=reused,
                            **timed(
                                lambda: c.hessenberg(
                                    a,
                                    q_backend="device",
                                    session=session if reused else None,
                                )
                            )
                        )
                    )
                    entry["paired_whole"].append(
                        dict(
                            label=label,
                            session=reused,
                            **timed(
                                lambda: c.right_eigenvectors(
                                    c.schur(
                                        a,
                                        qr_backend="device_control",
                                        q_backend="device",
                                        session=session if reused else None,
                                    )
                                )
                            )
                        )
                    )
                entry["host_after"] = timed(lambda: torch.linalg.eig(a))
                entry["session_uses"] = session.uses
            assert session.closed

            def cold_session():
                with c.GehrdSession() as cold:
                    return c.hessenberg(a, q_backend="device", session=cold)

            entry["cold_session_including_create_close"] = timed(cold_session)
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
        with c.GehrdSession() as session:
            c.hessenberg(a, q_backend="device", session=session)
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
