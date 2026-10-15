#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded native GPU Hessenberg/Schur prototype references and failure contract."""
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
    import gfx1013_schur_candidate as candidate
    torch.set_num_threads(4)
    torch.manual_seed(101323)
    arch = torch.cuda.get_device_properties(0).gcnArchName
    assert arch.split(':')[0] == 'gfx1013'
    report = dict(status='running', arch=arch, checks=[],
                  candidate_sha256=hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),
                  execution='GPU matrix arithmetic, host scalar convergence/control only')
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2)+'\n')

    def check(name, function):
        start = time.monotonic()
        try:
            values = function()
            torch.cuda.synchronize()
            report['checks'].append(dict(name=name, status='passed',
                                         seconds=time.monotonic()-start, **values))
            print('PASS', name, values, flush=True)
        except Exception as error:
            report['checks'].append(dict(name=name, status='failed', error=str(error)))
            print('FAIL', name, str(error), flush=True)
        save()

    def fixture(n, dtype, condition):
        q, _ = torch.linalg.qr(torch.randn(n,n,dtype=torch.complex128))
        r, _ = torch.linalg.qr(torch.randn(n,n,dtype=torch.complex128))
        scales = torch.linspace(1., condition, n, dtype=torch.float64)
        x = (q*scales)@r.mH
        w = torch.linspace(-2.,3.,n,dtype=torch.float64)+1j*torch.linspace(.3,1.7,n)
        return ((x*w)@torch.linalg.inv(x)).to(dtype), w

    def errors(a, h, q):
        a, h, q = [x.cpu().to(torch.complex128) for x in (a,h,q)]
        identity = torch.eye(a.shape[0], dtype=torch.complex128)
        return dict(similarity=float((a-q@h@q.mH).norm()/a.norm()),
                    orthogonality=float((q.mH@q-identity).norm()/a.shape[0]**.5))

    def run(n, dtype, condition):
        a, known = fixture(n,dtype,condition)
        tol = 5e-5 if dtype == torch.complex64 else 3e-12
        backing = torch.full((n+2,2*n+3), complex(311.,-197.),dtype=dtype,device='cuda')
        ga = backing[1:n+1,1:2*n+1:2].mT
        ga.copy_(a)
        before = backing.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            h,q = candidate.hessenberg(ga)
            result = candidate.schur(ga)
        torch.cuda.current_stream().wait_stream(stream)
        assert all(x.is_cuda for x in (h,q,result.t,result.q))
        torch.testing.assert_close(backing,before,rtol=0,atol=0)
        he = errors(a,h,q)
        se = errors(a,result.t,result.q)
        assert result.info == 0, (result.info,result.iterations)
        assert max(*he.values(),*se.values()) < tol,(he,se)
        assert float(torch.tril(h,-2).abs().max()) == 0
        assert float(torch.tril(result.t,-1).abs().max()) == 0
        actual = result.t.diag().cpu().to(torch.complex128)
        # CPU eig is reference-only, after the candidate has finished.
        reference = torch.linalg.eigvals(a.to(torch.complex128))
        costs = (actual[:,None]-reference[None,:]).abs()
        indices = costs.argmin(dim=1)
        assert indices.unique().numel() == n
        torch.testing.assert_close(actual,reference[indices],rtol=tol*5,atol=tol*5)
        analytic_costs = (actual[:,None]-known[None,:]).abs()
        assert float(analytic_costs.min(dim=1).values.max()) < tol*20
        return dict(hessenberg=he,schur=se,iterations=result.iterations,
                    deflations=result.deflations,eigenvalue_error=float(costs.min(dim=1).values.max()),
                    tolerance=tol,input_and_guards_unchanged=True,stream='nondefault')

    for dtype in (torch.complex64,torch.complex128):
        for n in (4,9,16):
            for condition in (1,10):
                check(f'schur_{dtype}_n{n}_basis{condition}',
                      lambda n=n,dtype=dtype,condition=condition:run(n,dtype,condition))

        def budget(dtype=dtype):
            a,_ = fixture(9,dtype,10)
            ga=a.cuda();before=ga.clone()
            attempts=[]
            for allowed in (0,1):
                result=candidate.schur(ga,max_iterations=allowed)
                assert result.info > 0 and result.iterations == allowed
                torch.testing.assert_close(ga,before,rtol=0,atol=0)
                bound=5e-5 if dtype==torch.complex64 else 3e-12
                metric=errors(a,result.t,result.q)
                assert max(metric.values()) < bound
                attempts.append(dict(info=result.info,iterations=result.iterations,similarity=metric))
            return dict(attempts=attempts,hidden_host_fallback=False)
        check(f'nonconvergence_{dtype}',budget)

        def triangular(dtype=dtype):
            a=torch.tensor([[1+2j,2-1j,1],[0,1+2j,3j],[0,0,-2+.5j]],dtype=dtype,device='cuda')
            result=candidate.schur(a,max_iterations=0)
            assert result.info == 0 and result.iterations == 0
            torch.testing.assert_close(result.t.diag(),a.diag())
            return dict(repeated_eigenvalues=True,iterations=0,limits='Already triangular; no defective eigvector claim')
        check(f'triangular_{dtype}',triangular)

        def rotation(dtype=dtype):
            a=torch.tensor([[0,1],[-1,0]],dtype=dtype,device='cuda')
            result=candidate.schur(a)
            assert result.info==0
            actual=result.t.diag().cpu().to(torch.complex128)
            expected=torch.tensor([1j,-1j],dtype=torch.complex128)
            cost=(actual[:,None]-expected[None,:]).abs()
            assert cost.argmin(dim=1).unique().numel()==2
            bound=5e-5 if dtype==torch.complex64 else 3e-12
            assert float(cost.min(dim=1).values.max())<bound
            metric=errors(a,result.t,result.q)
            assert max(metric.values())<bound
            return dict(iterations=result.iterations,nonreal_eigenvalues=True,similarity=metric)
        check(f'rotation_2x2_{dtype}',rotation)

        def scalar(dtype=dtype):
            a=torch.tensor([[2-3j]],dtype=dtype,device='cuda')
            result=candidate.schur(a,max_iterations=0)
            assert result.info==0 and result.iterations==0
            torch.testing.assert_close(result.t,a,rtol=0,atol=0)
            torch.testing.assert_close(result.q,torch.ones_like(a),rtol=0,atol=0)
            return dict(iterations=0)
        check(f'scalar_{dtype}',scalar)

    def invalid():
        fixtures=[torch.eye(17,dtype=torch.complex64,device='cuda'),
                  torch.eye(3,dtype=torch.float32,device='cuda'),
                  torch.ones(2,3,dtype=torch.complex64,device='cuda'),
                  torch.full((2,2),complex(float('nan'),0),dtype=torch.complex64,device='cuda'),
                  torch.eye(2,dtype=torch.complex64,device='cuda',requires_grad=True)]
        for a in fixtures:
            try:candidate.schur(a)
            except ValueError:continue
            raise AssertionError('Unsupported input accepted')
        a=torch.eye(2,dtype=torch.complex64,device='cuda')
        for budget in (-1,4097,True,1.5):
            try:candidate.schur(a,max_iterations=budget)
            except ValueError:continue
            raise AssertionError('Invalid budget accepted')
        return dict(rejections=len(fixtures)+4)
    check('argument_contract',invalid)
    report['mapped_solver_libraries']=sorted({line.split()[-1] for line in
        Path('/proc/self/maps').read_text().splitlines() if
        any(s in line for s in ('librocsolver','librocblas','libhipsolver'))})
    report['passed']=sum(x['status']=='passed' for x in report['checks'])
    report['status']='passed' if report['passed']==len(report['checks']) else 'failed'
    save()
    raise SystemExit(0 if report['status']=='passed' else 1)


if __name__=='__main__':main()
