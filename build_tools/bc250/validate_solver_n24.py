#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""N2.4 native small-Schur right-vector recovery and refusal references."""
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
    torch.manual_seed(101324)
    arch = torch.cuda.get_device_properties(0).gcnArchName
    assert arch.split(':')[0] == 'gfx1013'
    report = dict(status='running', arch=arch, checks=[],
                  candidate_sha256=hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),
                  execution='GPU matrix/vector arithmetic; host scalar control only')
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
        scales = torch.linspace(1.,condition,n,dtype=torch.float64)
        x = (q*scales)@r.mH
        w = torch.linspace(-2.,3.,n,dtype=torch.float64)+1j*torch.linspace(.3,1.7,n,dtype=torch.float64)
        a = ((x*w)@torch.linalg.inv(x)).to(dtype)
        return a,w,x/x.norm(dim=0)[None,:]

    def directions(actual, expected):
        error=0.
        for j in range(actual.shape[-1]):
            v,c=actual[:,j:j+1],expected[:,j:j+1]
            error=max(error,float((v@v.mH-c@c.mH).norm()))
        return error

    def run(n,dtype,condition):
        a,known,analytic_vectors=fixture(n,dtype,condition)
        backing=torch.full((n+2,2*n+3),complex(311.,-197.),dtype=dtype,device='cuda')
        ga=backing[1:n+1,1:2*n+1:2]
        if condition>1:ga=ga.mT
        ga.copy_(a); before=backing.clone()
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            schur=candidate.schur(ga)
            oldt,oldq=schur.t.clone(),schur.q.clone()
            result=candidate.right_eigenvectors(schur)
        torch.cuda.current_stream().wait_stream(stream)
        assert result.values.is_cuda and result.vectors.is_cuda
        torch.testing.assert_close(backing,before,rtol=0,atol=0)
        torch.testing.assert_close(schur.t,oldt,rtol=0,atol=0)
        torch.testing.assert_close(schur.q,oldq,rtol=0,atol=0)
        w=result.values.cpu().to(torch.complex128)
        v=result.vectors.cpu().to(torch.complex128)
        ref=a.to(torch.complex128)
        cw,cv=torch.linalg.eig(ref)
        costs=(w[:,None]-cw[None,:]).abs();indices=costs.argmin(dim=1)
        assert indices.unique().numel()==n
        tol=6e-4 if dtype==torch.complex64 else 4e-10
        torch.testing.assert_close(w,cw[indices],rtol=tol,atol=tol)
        torch.testing.assert_close(v.norm(dim=0),torch.ones(n,dtype=torch.float64),rtol=tol,atol=tol)
        residuals=(ref@v-v*w[None,:]).norm(dim=0)/((ref.norm()+w.abs())*v.norm(dim=0))
        residual=float(residuals.max())
        bound=8e-6 if dtype==torch.complex64 else 3e-12
        assert residual<bound,residual
        cpu_direction=directions(v,cv[:,indices])
        analytic_cost=(w[:,None]-known[None,:]).abs();ai=analytic_cost.argmin(dim=1)
        assert ai.unique().numel()==n
        torch.testing.assert_close(w,known[ai],rtol=tol,atol=tol)
        analytic_direction=directions(v,analytic_vectors[:,ai])
        assert max(cpu_direction,analytic_direction)<tol,(cpu_direction,analytic_direction)
        return dict(order=n,basis_condition=condition,iterations=schur.iterations,
                    eigenpair_residual=residual,residual_bound=bound,cpu_projector_error=cpu_direction,
                    analytic_projector_error=analytic_direction,direction_bound=tol,
                    relative_separation=result.relative_separation,
                    triangular_residual=result.triangular_residual,
                    input_and_schur_unchanged=True,stream='nondefault')

    def form(t,q=None,info=0):
        if q is None:q=torch.eye(t.shape[0],dtype=t.dtype,device=t.device)
        return candidate.SchurResult(t,q,info,0,0)

    def expect_refusal(result,kind,text):
        t,q=result.t.clone(),result.q.clone()
        try:candidate.right_eigenvectors(result)
        except kind as error:
            assert text in str(error),(str(error),text)
        else:raise AssertionError('Unsupported recovery accepted')
        torch.testing.assert_close(result.t,t,rtol=0,atol=0,equal_nan=True)
        torch.testing.assert_close(result.q,q,rtol=0,atol=0,equal_nan=True)

    for dtype in (torch.complex64,torch.complex128):
        for n in (4,9,16):
            for condition in (1,10):
                check(f'eigvectors_{dtype}_n{n}_basis{condition}',
                      lambda n=n,dtype=dtype,condition=condition:run(n,dtype,condition))

        def analytic(dtype=dtype):
            t=torch.tensor([[1,4],[0,2+1j]],dtype=dtype,device='cuda')
            result=candidate.right_eigenvectors(form(t))
            expected=torch.tensor([[1,4/(1+1j)],[0,1]],dtype=torch.complex128)
            expected=expected/expected.norm(dim=0)[None,:]
            error=directions(result.vectors.cpu().to(torch.complex128),expected)
            assert error<(6e-5 if dtype==torch.complex64 else 3e-12)
            # Identity Schur vectors are deliberately not eigenvectors here.
            assert float((result.vectors.mH@result.vectors-torch.eye(2,device='cuda')).norm())>.5
            return dict(projector_error=error,schur_vectors_are_not_eigenvectors=True)
        check(f'analytic_nonnormal_2x2_{dtype}',analytic)

        def separation(dtype=dtype):
            floor=8*torch.finfo(torch.empty((),dtype=dtype).real.dtype).eps**.5
            tests=[torch.tensor([[1,0],[0,1]],dtype=dtype,device='cuda'),
                   torch.tensor([[1,1],[0,1]],dtype=dtype,device='cuda'),
                   torch.tensor([[1,0],[0,1+floor/2]],dtype=dtype,device='cuda'),
                   torch.zeros(2,2,dtype=dtype,device='cuda')]
            for t in tests:expect_refusal(form(t),RuntimeError,'separated')
            t=torch.diag(torch.tensor([1,1+4*floor],dtype=dtype,device='cuda'))
            result=candidate.right_eigenvectors(form(t))
            torch.testing.assert_close(result.vectors,torch.eye(2,dtype=dtype,device='cuda'),rtol=0,atol=0)
            return dict(refusals=4,relative_separation_floor=floor,above_floor_control=True)
        check(f'separation_contract_{dtype}',separation)

        def unconverged(dtype=dtype):
            a,_,_=fixture(9,dtype,10)
            for budget in (0,1):
                partial=candidate.schur(a.cuda(),max_iterations=budget)
                assert partial.info>0
                expect_refusal(partial,RuntimeError,'unconverged')
            return dict(refusals=2,budgets=[0,1])
        check(f'unconverged_contract_{dtype}',unconverged)

        def scalar(dtype=dtype):
            for value in (0,2-3j):
                t=torch.tensor([[value]],dtype=dtype,device='cuda')
                result=candidate.right_eigenvectors(candidate.schur(t,max_iterations=0))
                torch.testing.assert_close(result.values,t.diag(),rtol=0,atol=0)
                torch.testing.assert_close(result.vectors,torch.ones_like(t),rtol=0,atol=0)
            return dict(scalars=2)
        check(f'scalar_{dtype}',scalar)

    def invalid():
        t=torch.diag(torch.tensor([1,2+1j],dtype=torch.complex64,device='cuda'))
        badlower=t.clone();badlower[1,0]=.01
        expect_refusal(form(badlower),ValueError,'upper triangular')
        expect_refusal(form(t,2*torch.eye(2,dtype=t.dtype,device=t.device)),ValueError,'unitary')
        expect_refusal(form(t,torch.eye(3,dtype=t.dtype,device=t.device)),ValueError,'matching')
        expect_refusal(form(t,torch.eye(2,dtype=torch.complex128,device=t.device)),ValueError,'matching')
        badq=torch.eye(2,dtype=t.dtype,device=t.device);badq[0,0]=complex(float('nan'),0)
        expect_refusal(form(t,badq),ValueError,'finite')
        gradq=torch.eye(2,dtype=t.dtype,device=t.device,requires_grad=True)
        expect_refusal(form(t,gradq),ValueError,'autograd')
        try:candidate.right_eigenvectors(None)
        except ValueError:pass
        else:raise AssertionError('Missing result accepted')
        return dict(rejections=7)
    check('invalid_schur_contract',invalid)
    report['mapped_solver_libraries']=sorted({line.split()[-1] for line in
        Path('/proc/self/maps').read_text().splitlines() if
        any(s in line for s in ('librocsolver','librocblas','libhipsolver'))})
    report['passed']=sum(x['status']=='passed' for x in report['checks'])
    report['status']='passed' if report['passed']==len(report['checks']) else 'failed'
    report['limits']='Isolated order1..16 separated-spectrum prototype, no production eig integration, balancing/batching/gradient/performance claim'
    save()
    raise SystemExit(0 if report['status']=='passed' else 1)


if __name__=='__main__':main()
