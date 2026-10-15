#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded nonsymmetric complex eig compatibility checks, including host LAPACK."""
import argparse
import hashlib
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(101322)
    assert torch.cuda.is_available()
    arch = torch.cuda.get_device_properties(0).gcnArchName
    assert arch.split(':')[0] == 'gfx1013'
    report = dict(status='running', arch=arch, torch=torch.__version__,
                  hip=torch.version.hip, rocm=torch.version.rocm, checks=[],
                  execution='GPU tensor interface; pinned hipSOLVER Xgeev uses CPU LAPACK')
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2)+'\n')

    def check(name, function):
        start = time.monotonic()
        try:
            metrics = function()
            torch.cuda.synchronize()
            report['checks'].append(dict(name=name, status='passed',
                                         seconds=time.monotonic()-start, **metrics))
            print('PASS', name, metrics, flush=True)
        except Exception as error:
            report['checks'].append(dict(name=name, status='failed', error=str(error)))
            print('FAIL', name, str(error), flush=True)
        save()

    def fixture(n, condition, dtype):
        q, _ = torch.linalg.qr(torch.randn(2, n, n, dtype=torch.complex128))
        r, _ = torch.linalg.qr(torch.randn(2, n, n, dtype=torch.complex128))
        scales = torch.logspace(0, torch.log10(torch.tensor(float(condition))).item(),
                                n, dtype=torch.float64)
        x = (q*scales)@r.mH
        # Strictly ordered real parts make matching unambiguous; not an API ordering assumption.
        values = torch.linspace(-3., 4., n, dtype=torch.float64) + \
                 1j*(torch.sin(torch.arange(n, dtype=torch.float64))+.2)
        a = ((x*values)@torch.linalg.inv(x)).to(dtype)
        return a, values.expand(2, n)

    def eigenpairs(n, condition, dtype, layout):
        a, known = fixture(n, condition, dtype)
        ref = a.to(torch.complex128)
        cw, cv = torch.linalg.eig(ref)
        order = cw.real.argsort(dim=-1)
        cw = cw.gather(-1, order)
        cv = cv.gather(-1, order[:, None, :].expand_as(cv))
        assert bool((cw.real.diff(dim=-1) > .05).all())
        tol = 4e-4 if dtype == torch.complex64 else 2e-10
        torch.testing.assert_close(cw, known, rtol=tol, atol=tol)
        backing = torch.full((2, n+2, 2*n+3), complex(311., -197.),
                             dtype=dtype, device='cuda')
        ga = backing[:, 1:n+1, 1:2*n+1:2]
        if layout == 'transposed':
            ga = ga.mT
        ga.copy_(a)
        before = backing.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            w, v = torch.linalg.eig(ga)
            only = torch.linalg.eigvals(ga)
            assert w.is_cuda and v.is_cuda and only.is_cuda
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(backing, before, rtol=0, atol=0)
        w, v, only = w.cpu().to(torch.complex128), v.cpu().to(torch.complex128), \
                     only.cpu().to(torch.complex128)
        order = w.real.argsort(dim=-1)
        w = w.gather(-1, order)
        v = v.gather(-1, order[:, None, :].expand_as(v))
        only = only.gather(-1, only.real.argsort(dim=-1))
        torch.testing.assert_close(w, cw, rtol=tol, atol=tol)
        torch.testing.assert_close(only, cw, rtol=tol, atol=tol)
        residual = float(((ref@v-v*w[:, None, :]).norm(dim=(-2,-1)) /
                          (ref.norm(dim=(-2,-1))*v.norm(dim=(-2,-1)))).max())
        residual_bound = 8e-6 if dtype == torch.complex64 else 2e-12
        assert residual < residual_bound, residual
        norms = v.norm(dim=-2)
        torch.testing.assert_close(norms, torch.ones_like(norms), rtol=tol, atol=tol)
        projector = 0.
        for i in range(n):
            vi, ci = v[..., i:i+1], cv[..., i:i+1]
            projector = max(projector, float((vi@vi.mH-ci@ci.mH).norm(dim=(-2,-1)).max()))
        assert projector < tol*10, projector
        assert float((ref-ref.mH).norm()) > 1., 'Nonsymmetric fixture required'
        return dict(shape=list(ga.shape), stride=list(ga.stride()), basis_condition=condition,
                    eigenvalue_error=float((w-cw).abs().max()), residual=residual,
                    residual_bound=residual_bound, projector_error=projector,
                    reference_tolerance=tol, input_and_guards_unchanged=True,
                    stream='nondefault', eigvals_compared=True)

    for dtype in (torch.complex64, torch.complex128):
        for n in (9, 33, 65):
            for condition in (1, 20):
                layout = 'transposed' if condition == 20 else 'padded'
                check(f'eig_{dtype}_n{n}_basis{condition}_{layout}',
                      lambda n=n, condition=condition, dtype=dtype, layout=layout:
                      eigenpairs(n, condition, dtype, layout))

        def gradient(dtype=dtype):
            a, _ = fixture(9, 3, dtype)
            cpu = a.to(torch.complex128).clone().requires_grad_()
            gpu = a.cuda().requires_grad_()
            for matrix in (cpu, gpu):
                w, _ = torch.linalg.eig(matrix)
                # Symmetric eigenvalue objective does not assume backend ordering or vector phase.
                w.abs().square().mean().backward()
            tol = 6e-4 if dtype == torch.complex64 else 3e-10
            actual = gpu.grad.cpu().to(torch.complex128)
            torch.testing.assert_close(actual, cpu.grad, rtol=tol, atol=tol)
            return dict(relative_gradient_error=float((actual-cpu.grad).norm()/cpu.grad.norm()),
                        tolerance=tol)
        check(f'eig_backward_{dtype}', gradient)

        def empty(dtype=dtype):
            for shape in ((0, 0), (0, 3, 3), (2, 0, 0)):
                a = torch.empty(shape, dtype=dtype, device='cuda')
                w, v = torch.linalg.eig(a)
                assert w.shape == a.shape[:-1] and v.shape == a.shape
                assert w.is_cuda and v.is_cuda
            return dict(shapes=[[0,0],[0,3,3],[2,0,0]], limits='Torch empty fast path; no LAPACK execution')
        check(f'eig_empty_{dtype}', empty)

    maps = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
            if any(s in line for s in ('libhipsolver', 'liblapack', 'libopenblas'))}
    report['mapped_libraries'] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                  for p in sorted(maps) if Path(p).is_file()}
    report['passed'] = sum(c['status']=='passed' for c in report['checks'])
    report['status'] = 'passed' if report['passed']==len(report['checks']) else 'failed'
    report['limits'] = ('Host-LAPACK compatibility only; no native GPU nonsymmetric eigensolver, '
                        'defective/repeated/extreme-scale/nonfinite/convergence-failure or performance claim')
    save()
    raise SystemExit(0 if report['status']=='passed' else 1)


if __name__ == '__main__':
    main()
