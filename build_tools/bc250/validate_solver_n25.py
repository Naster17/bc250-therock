#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded normal-scale and separation boundaries for the native eig candidate."""
import argparse
import hashlib
import json
import time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--group',choices=('baseline','scale','boundary','failure'),default='scale')
    args=parser.parse_args()
    import torch
    import gfx1013_schur_candidate as candidate
    torch.set_num_threads(4);torch.manual_seed(101325)
    assert torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]=='gfx1013'
    report=dict(status='running',group=args.group,checks=[],
                candidate_sha256=hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():args.output.write_text(json.dumps(report,indent=2)+'\n')
    def check(name,fn):
        start=time.monotonic()
        try:
            metrics=fn();torch.cuda.synchronize()
            report['checks'].append(dict(name=name,status='passed',seconds=time.monotonic()-start,**metrics))
            print('PASS',name,metrics,flush=True)
        except Exception as error:
            report['checks'].append(dict(name=name,status='failed',error=str(error)))
            print('FAIL',name,str(error),flush=True)
        save()
    def fixture(n,condition):
        q,_=torch.linalg.qr(torch.randn(n,n,dtype=torch.complex128))
        r,_=torch.linalg.qr(torch.randn(n,n,dtype=torch.complex128))
        x=(q*torch.linspace(1.,condition,n,dtype=torch.float64))@r.mH
        w=torch.linspace(-2.,3.,n,dtype=torch.float64)+1j*torch.linspace(.3,1.7,n,dtype=torch.float64)
        return (x*w)@torch.linalg.inv(x),w,x/x.norm(dim=0)[None,:]
    def direction_error(actual,expected):
        return max(float((actual[:,j:j+1]@actual[:,j:j+1].mH-
                          expected[:,j:j+1]@expected[:,j:j+1].mH).norm())
                   for j in range(actual.shape[1]))
    def scale_case(dtype,n,condition,scale,base,known,analytic):
        a=(base*scale).to(dtype)
        # Normalize references AFTER input quantization, before any norms/eig.
        ref=a.to(torch.complex128)/scale
        backing=torch.full((n+2,2*n+3),complex(311.,-197.),dtype=dtype,device='cuda')
        ga=backing[1:n+1,1:2*n+1:2].mT;ga.copy_(a);before=backing.clone()
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            schur=candidate.schur(ga)
            assert schur.info==0,(schur.info,schur.iterations)
            result=candidate.right_eigenvectors(schur)
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(backing,before,rtol=0,atol=0)
        assert result.values.is_cuda and result.vectors.is_cuda
        w=result.values.cpu().to(torch.complex128)/scale
        v=result.vectors.cpu().to(torch.complex128)
        t=schur.t.cpu().to(torch.complex128)/scale;q=schur.q.cpu().to(torch.complex128)
        cw,cv=torch.linalg.eig(ref)
        cost=(w[:,None]-cw[None,:]).abs();indices=cost.argmin(dim=1)
        assert indices.unique().numel()==n
        tolerance=8e-4 if dtype==torch.complex64 else 6e-10
        torch.testing.assert_close(w,cw[indices],rtol=tolerance,atol=tolerance)
        analytic_indices=(w[:,None]-known[None,:]).abs().argmin(dim=1)
        assert analytic_indices.unique().numel()==n
        torch.testing.assert_close(w,known[analytic_indices],rtol=tolerance,atol=tolerance)
        residual=float(((ref@v-v*w[None,:]).norm(dim=0)/((ref.norm()+w.abs())*v.norm(dim=0))).max())
        similarity=float((ref-q@t@q.mH).norm()/ref.norm())
        bound=8e-6 if dtype==torch.complex64 else 4e-12
        assert max(residual,similarity)<bound,(residual,similarity)
        cpu_error=direction_error(v,cv[:,indices]);analytic_error=direction_error(v,analytic[:,analytic_indices])
        assert max(cpu_error,analytic_error)<tolerance,(cpu_error,analytic_error)
        return dict(scale=scale,order=n,basis_condition=condition,iterations=schur.iterations,
                    residual=residual,similarity=similarity,residual_bound=bound,
                    cpu_projector_error=cpu_error,analytic_projector_error=analytic_error,
                    direction_bound=tolerance,guards_unchanged=True,stream='nondefault')
    for dtype in (torch.complex64,torch.complex128):
        exponents=(-25,-10,0,10,25) if dtype==torch.complex64 else (-200,-80,0,80,200)
        if args.group=='baseline':exponents=(exponents[0],0,exponents[-1])
        if args.group in ('scale','baseline'):
            shapes=((4,10),) if args.group=='baseline' else ((4,1),(4,10),(9,1),(9,10))
            for n,condition in shapes:
                base,known,analytic=fixture(n,condition)
                for power in exponents:
                    check(f'scale_{dtype}_n{n}_basis{condition}_e{power}',
                          lambda dtype=dtype,n=n,condition=condition,power=power,base=base,known=known,analytic=analytic:
                          scale_case(dtype,n,condition,10.**power,base,known,analytic))
        if args.group=='boundary':
            for power in (exponents[0],0,exponents[-1]):
                for factor in (.5,.9,1.1,2.,4.):
                    def boundary(dtype=dtype,power=power,factor=factor):
                        eps=torch.finfo(torch.empty((),dtype=dtype).real.dtype).eps
                        floor=8*eps**.5;scale=10.**power
                        raw=torch.tensor([[1+.2j,.3],[0,1+factor*floor+.2j]],dtype=torch.complex128)
                        a=(raw*scale).to(dtype);ref=a.to(torch.complex128)/scale
                        separation=float((ref[0,0]-ref[1,1]).abs()/ref.abs().max())
                        expect_accept=separation>floor
                        t=a.cuda();q=torch.eye(2,dtype=dtype,device='cuda')
                        before=t.clone();form=candidate.SchurResult(t,q,0,0,0)
                        try:result=candidate.right_eigenvectors(form)
                        except RuntimeError as error:
                            assert not expect_accept,(separation,floor,str(error))
                            assert 'separated' in str(error) or 'pivot' in str(error)
                            accepted=False
                        else:
                            assert expect_accept,(separation,floor)
                            accepted=True
                            expected=torch.tensor([[1,0],[0,1]],dtype=torch.complex128)
                            expected[0,1]=ref[0,1]/(ref[1,1]-ref[0,0]);expected/=expected.norm(dim=0)[None,:]
                            error=direction_error(result.vectors.cpu().to(torch.complex128),expected)
                            assert error<(8e-4 if dtype==torch.complex64 else 6e-10),error
                        torch.testing.assert_close(t,before,rtol=0,atol=0)
                        return dict(scale=scale,factor=factor,relative_separation=separation,floor=floor,accepted=accepted)
                    check(f'boundary_{dtype}_e{power}_gap{factor}',boundary)
    if args.group=='failure':
        for dtype in (torch.complex64,torch.complex128):
            extremes=(-25,25) if dtype==torch.complex64 else (-200,200)
            for power in extremes:
                for budget in (0,1):
                    def partial(dtype=dtype,power=power,budget=budget):
                        base,_,_=fixture(4,10);scale=10.**power
                        a=(base*scale).to(dtype);ga=a.cuda();before=ga.clone()
                        result=candidate.schur(ga,max_iterations=budget)
                        assert result.info>0 and result.iterations==budget
                        q=result.q.cpu().to(torch.complex128)
                        t=result.t.cpu().to(torch.complex128)/scale
                        ref=a.to(torch.complex128)/scale
                        error=float((ref-q@t@q.mH).norm()/ref.norm())
                        assert error<(8e-6 if dtype==torch.complex64 else 4e-12),error
                        try:candidate.right_eigenvectors(result)
                        except RuntimeError as e:assert 'unconverged' in str(e)
                        else:raise AssertionError('Unconverged result accepted')
                        torch.testing.assert_close(ga,before,rtol=0,atol=0)
                        return dict(scale=scale,budget=budget,info=result.info,similarity=error)
                    check(f'partial_{dtype}_e{power}_budget{budget}',partial)
            def overflow(dtype=dtype):
                real=torch.empty((),dtype=dtype).real.dtype
                a=torch.full((2,2),complex(.75*torch.finfo(real).max,0),dtype=dtype,device='cuda')
                before=a.clone()
                try:candidate.schur(a)
                except RuntimeError as e:assert 'rescaling' in str(e),str(e)
                else:raise AssertionError('Unrepresentable Schur output accepted')
                torch.testing.assert_close(a,before,rtol=0,atol=0)
                return dict(refused='finite input has nonrepresentable Schur form',input_unchanged=True)
            check(f'rescale_overflow_{dtype}',overflow)
            def subnormal(dtype=dtype):
                real=torch.empty((),dtype=dtype).real.dtype
                integer=torch.int32 if dtype==torch.complex64 else torch.int64
                tiny=torch.finfo(real).tiny
                cpu=torch.diag(torch.tensor([tiny/8,tiny/4],dtype=real)).to(dtype)
                a=cpu.cuda();before=torch.view_as_real(a).view(integer).clone()
                try:candidate.schur(a)
                except RuntimeError as e:assert 'scale unsupported' in str(e),str(e)
                else:raise AssertionError('All-subnormal input accepted')
                assert torch.equal(torch.view_as_real(a).view(integer),before)
                assert torch.equal(before.cpu(),torch.view_as_real(cpu).view(integer))
                return dict(refused='all-subnormal scale',input_bits_unchanged=True)
            check(f'subnormal_refusal_{dtype}',subnormal)
            def zero(dtype=dtype):
                a=torch.zeros(4,4,dtype=dtype,device='cuda')
                result=candidate.schur(a,max_iterations=0)
                assert result.info==0 and result.iterations==0
                torch.testing.assert_close(result.t,a,rtol=0,atol=0)
                try:candidate.right_eigenvectors(result)
                except RuntimeError as e:assert 'separated' in str(e)
                else:raise AssertionError('Repeated zero spectrum accepted')
                return dict(schur_converged=True,eigenvectors_refused=True)
            check(f'zero_{dtype}',zero)
            def views(dtype=dtype):
                base,_,_=fixture(4,3);source=base.to(dtype).cuda()
                errors=[]
                for a in (source.conj(),source._neg_view()):
                    before=a.clone();result=candidate.right_eigenvectors(candidate.schur(a))
                    ref=a.cpu().to(torch.complex128)
                    v=result.vectors.cpu().to(torch.complex128);w=result.values.cpu().to(torch.complex128)
                    error=float(((ref@v-v*w[None,:]).norm(dim=0)/((ref.norm()+w.abs())*v.norm(dim=0))).max())
                    assert error<(8e-6 if dtype==torch.complex64 else 4e-12),error
                    torch.testing.assert_close(a,before,rtol=0,atol=0);errors.append(error)
                return dict(conjugate_and_negative_views=True,residuals=errors)
            check(f'view_controls_{dtype}',views)
    report['passed']=sum(x['status']=='passed' for x in report['checks'])
    report['status']='passed' if report['passed']==len(report['checks']) else 'failed'
    save()
    raise SystemExit(0 if report['status']=='passed' else 1)


if __name__=='__main__':main()
