#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded synchronized wall cost and ATen scalar-read profile, not kernel timing."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--before',type=Path,required=True)
    ap.add_argument('--after',type=Path)
    args=ap.parse_args()
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    torch.set_num_threads(4);torch.manual_seed(101326)
    assert torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]=='gfx1013'
    before=load(args.before,'schur_n26_before')
    after=load(args.after,'schur_n26_after') if args.after else None
    report=dict(status='running',fixtures=[],
                before_sha256=hashlib.sha256(args.before.read_bytes()).hexdigest(),
                after_sha256=hashlib.sha256(args.after.read_bytes()).hexdigest() if args.after else None,
                method='Wall time with device synchronization before/after each call;3 samples/pass',
                limits='Six small fixtures; ATen scalar-read counts are not full HIP activity traces')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():args.output.write_text(json.dumps(report,indent=2)+'\n')
    def timed(fn):
        fn();torch.cuda.synchronize()
        samples=[]
        for _ in range(3):
            torch.cuda.synchronize();start=time.perf_counter()
            result=fn();torch.cuda.synchronize()
            samples.append((time.perf_counter()-start)*1000)
            del result
        return dict(median_ms=statistics.median(samples),samples_ms=samples)
    class Counts(TorchDispatchMode):
        def __init__(self):super().__init__();self.counts=Counter()
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            self.counts[str(func)]+=1
            return func(*args,**(kwargs or {}))
    def profile(fn):
        with Counts() as counts:fn()
        torch.cuda.synchronize()
        return dict(scalar_reads=counts.counts.get('aten._local_scalar_dense.default',0),
                    aten_calls=sum(counts.counts.values()),operators=dict(counts.counts.most_common(15)))
    def fixture(n,dtype):
        q,_=torch.linalg.qr(torch.randn(n,n,dtype=torch.complex128))
        r,_=torch.linalg.qr(torch.randn(n,n,dtype=torch.complex128))
        x=(q*torch.linspace(1.,10.,n,dtype=torch.float64))@r.mH
        w=torch.linspace(-2.,3.,n,dtype=torch.float64)+1j*torch.linspace(.3,1.7,n,dtype=torch.float64)
        return ((x*w)@torch.linalg.inv(x)).to(dtype)
    def verify(a,values,vectors):
        ref=a.to(torch.complex128)
        w=values.cpu().to(torch.complex128);v=vectors.cpu().to(torch.complex128)
        cw,cv=torch.linalg.eig(ref)
        index=(w[:,None]-cw[None,:]).abs().argmin(dim=1)
        assert index.unique().numel()==a.shape[0]
        tol=8e-4 if a.dtype==torch.complex64 else 6e-10
        torch.testing.assert_close(w,cw[index],rtol=tol,atol=tol)
        residual=float(((ref@v-v*w[None,:]).norm(dim=0)/((ref.norm()+w.abs())*v.norm(dim=0))).max())
        bound=8e-6 if a.dtype==torch.complex64 else 4e-12
        assert residual<bound,residual
        direction=max(float((v[:,i:i+1]@v[:,i:i+1].mH-cv[:,index[i:i+1]]@cv[:,index[i:i+1]].mH).norm())
                      for i in range(a.shape[0]))
        assert direction<tol,direction
        return dict(residual=residual,projector_error=direction)
    for dtype in (torch.complex64,torch.complex128):
        for n in (4,9,16):
            a=fixture(n,dtype);ga=a.cuda();input_before=ga.clone()
            schur=before.schur(ga);assert schur.info==0
            def whole(module):return module.right_eigenvectors(module.schur(ga))
            stages={'schur':lambda:before.schur(ga),
                    'recovery':lambda:before.right_eigenvectors(schur),
                    'whole':lambda:whole(before),
                    'host_eig':lambda:torch.linalg.eig(ga)}
            baseline=whole(before);host=torch.linalg.eig(ga)
            entry=dict(dtype=str(dtype),order=n,iterations=schur.iterations,
                       correctness_before=verify(a,baseline.values,baseline.vectors),
                       correctness_host=verify(a,*host),components={},profile={})
            for stage in stages:
                entry['components'][stage]=timed(stages[stage])
            # Instrumentation is separate from timing; component times overlap.
            for stage in ('schur','recovery','host_eig'):
                entry['profile'][stage]=profile(stages[stage])
            if after:
                other_form=after.SchurResult(schur.t,schur.q,schur.info,schur.iterations,schur.deflations)
                proposed=whole(after)
                entry['correctness_after']=verify(a,proposed.values,proposed.vectors)
                entry['profile_after_recovery']=profile(lambda:after.right_eigenvectors(other_form))
                entry['paired']={}
                for stage in ('recovery','whole'):
                    passes=[]
                    for label,module,form in [('A1',before,schur),('B1',after,other_form),
                                               ('B2',after,other_form),('A2',before,schur)]:
                        fn=(lambda module=module,form=form:module.right_eigenvectors(form)) if stage=='recovery' else \
                           (lambda module=module:whole(module))
                        passes.append(dict(label=label,**timed(fn)))
                    ac=statistics.mean(passes[i]['median_ms'] for i in (0,3))
                    bc=statistics.mean(passes[i]['median_ms'] for i in (1,2))
                    entry['paired'][stage]=dict(before_over_after=ac/bc,
                        before_drift=passes[3]['median_ms']/passes[0]['median_ms'],passes=passes)
                entry['host_control_after']=timed(stages['host_eig'])
            torch.testing.assert_close(ga,input_before,rtol=0,atol=0)
            report['fixtures'].append(entry);save()
            print('SOLVER_PROFILE',str(dtype),n,
                  {k:round(v['median_ms'],3) for k,v in entry['components'].items()},
                  {k:v['scalar_reads'] for k,v in entry['profile'].items()},
                  {k:round(v['before_over_after'],3) for k,v in entry.get('paired',{}).items()},flush=True)
    report['status']='passed';save()


if __name__=='__main__':main()
