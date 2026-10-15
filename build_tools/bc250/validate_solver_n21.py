#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""N2.1: complex Hermitian eigenvectors, triangle contracts and stable subspaces."""
import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(101321)
    assert torch.cuda.is_available()
    arch = torch.cuda.get_device_properties(0).gcnArchName
    assert arch.split(':')[0] == 'gfx1013', arch
    report = dict(status='running', arch=arch, torch=torch.__version__,
                  backend=str(torch.backends.cuda.preferred_linalg_library()), checks=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + '\n')

    def record(name, function):
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

    def fixture(n, dtype, clustered):
        raw = torch.randn(2, n, n, dtype=torch.complex128)
        q, _ = torch.linalg.qr(raw)
        if clustered:
            cuts = [0, n//3, 2*n//3, n]
            spectrum = torch.cat([torch.full((cuts[i+1]-cuts[i],), value,
                                           dtype=torch.float64)
                                  for i, value in enumerate((-3., 0., 2.))])
        else:
            cuts = list(range(n+1))
            spectrum = torch.linspace(-4., 5., n, dtype=torch.float64)
        a = ((q*spectrum) @ q.mH).to(dtype)
        # Round once before the high-precision reference; make the Hermitian input exact.
        a = (a+a.mH)/2
        return a, cuts

    def eigenvectors(n, dtype, uplo, clustered):
        a, cuts = fixture(n, dtype, clustered)
        reference = a.to(torch.complex128)
        cw, cv = torch.linalg.eigh(reference, UPLO=uplo)
        tolerance = 8e-5 if dtype == torch.complex64 else 2e-11
        sentinel = complex(311., -197.)
        # Padded strided batch view; all guard lanes must survive the public call.
        backing = torch.full((2, n+2, 2*n+3), sentinel, dtype=dtype, device='cuda')
        ga = backing[:, 1:n+1, 1:2*n+1:2]
        assert not ga.is_contiguous()
        ga.copy_(a)
        ignore = torch.triu(torch.ones(n, n, dtype=torch.bool), 1) if uplo == 'L' else \
                 torch.tril(torch.ones(n, n, dtype=torch.bool), -1)
        ga[:, ignore] = complex(float('nan'), float('nan'))
        before = backing.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            w, v = torch.linalg.eigh(ga, UPLO=uplo)
            assert w.is_cuda and v.is_cuda
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(backing, before, rtol=0, atol=0, equal_nan=True)
        w, v = w.cpu().double(), v.cpu().to(torch.complex128)
        torch.testing.assert_close(w, cw, rtol=tolerance, atol=tolerance)
        identity = torch.eye(n, dtype=torch.complex128).expand(2, n, n)
        residual = float(((reference@v-v*w.unsqueeze(-2)).norm(dim=(-2, -1)) /
                          reference.norm(dim=(-2, -1))).max())
        orthogonality = float((v.mH@v-identity).norm(dim=(-2, -1)).max()/n**0.5)
        reconstruction = float(((v*w.unsqueeze(-2))@v.mH-reference).norm(dim=(-2,-1)).max() /
                               reference.norm(dim=(-2,-1)).max())
        assert residual < tolerance, residual
        assert orthogonality < tolerance, orthogonality
        assert reconstruction < tolerance, reconstruction
        projector = 0.
        for lo, hi in zip(cuts, cuts[1:]):
            actual = v[..., lo:hi]@v[..., lo:hi].mH
            expected = cv[..., lo:hi]@cv[..., lo:hi].mH
            error = float((actual-expected).norm(dim=(-2,-1)).max() / (hi-lo)**0.5)
            projector = max(projector, error)
        assert projector < tolerance*4, projector
        return dict(residual=residual, orthogonality=orthogonality,
                    reconstruction=reconstruction, projector=projector,
                    tolerance=tolerance, shape=list(ga.shape), input_stride=list(ga.stride()),
                    ignored_triangle_nan=True, stream='nondefault', input_and_guards_unchanged=True)

    for dtype in (torch.complex64, torch.complex128):
        for n, clustered in ((17, False), (65, False), (129, False), (257, False), (33, True)):
            for uplo in ('L', 'U'):
                record(f'eigh_{dtype}_n{n}_{uplo}_clustered{clustered}',
                       lambda n=n, dtype=dtype, uplo=uplo, clustered=clustered:
                       eigenvectors(n, dtype, uplo, clustered))

        def gradient(dtype=dtype):
            a, _ = fixture(11, dtype, False)
            cpu = a.to(torch.complex128).clone().requires_grad_()
            gpu = a.cuda().requires_grad_()
            for x in (cpu, gpu):
                w, v = torch.linalg.eigh(x)
                # Eigenvector phase-invariant objective; distinct eigenvalues only.
                weights = torch.linspace(.2, 1.1, 11, device=x.device, dtype=w.dtype)
                ((w.square()*weights).mean() +
                 (v.abs().square()*weights[:, None]*weights.square()[None, :]).mean()).backward()
            tolerance = 1e-3 if dtype == torch.complex64 else 5e-10
            torch.testing.assert_close(gpu.grad.cpu().to(torch.complex128), cpu.grad,
                                       rtol=tolerance, atol=tolerance)
            error = float((gpu.grad.cpu().to(torch.complex128)-cpu.grad).norm()/cpu.grad.norm())
            return dict(relative_gradient_error=error, tolerance=tolerance)
        record(f'eigh_backward_{dtype}', gradient)

    # Record the mapped runtime libraries rather than infer their identity from names.
    report['mapped_solver_libraries'] = sorted({line.split()[-1] for line in
        Path('/proc/self/maps').read_text().splitlines() if
        any(name in line for name in ('libhipsolver', 'librocsolver', 'librocblas'))})
    report['passed'] = sum(c['status']=='passed' for c in report['checks'])
    report['status'] = 'passed' if report['passed']==len(report['checks']) else 'failed'
    report['limits'] = ('Complex Hermitian eigh only, finite used triangle, two-matrix batches up to257; '
                        'ignored triangle NaNs, first-order gradient only on separated spectrum; '
                        'no non-Hermitian eig/SVD/large-scale/conditioning/performance claim')
    save()
    raise SystemExit(0 if report['status']=='passed' else 1)


if __name__ == '__main__':
    main()
