#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded full-model AMP, exact-state fresh-process resume, compiled math attention."""
import argparse
import copy
import json
import importlib.util
from pathlib import Path
import random
import subprocess
import sys

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--phase',choices=['amp','compile','train-child','resume-child'],required=True)
    ap.add_argument('--dtype',choices=['float32','float16','bfloat16'],default='float32')
    ap.add_argument('--backend',choices=['math','tiled'],default='math')
    ap.add_argument('--only-tiled',action='store_true')
    ap.add_argument('--candidate',type=Path)
    args=ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(720);random.seed(720)
    dtype=getattr(torch,args.dtype)
    report={'status':'running','checks':[]}
    def save():args.output.write_text(json.dumps(report,indent=2)+'\n')
    def passed(name,**data):
        report['checks'].append(dict(name=name,**data));save()
        print('TRAINING_EXPANSION_PASS',report['checks'][-1],flush=True)

    if args.phase=='amp':
        for dt,backend in [('float32','math'),('float16','math'),('bfloat16','math'),('bfloat16','tiled')]:
            if args.only_tiled and backend!='tiled':continue
            root=args.output.parent/(dt+'-'+backend);root.mkdir()
            for phase in ['train-child','resume-child']:
                command=[sys.executable,__file__,'--output',str(root/(phase+'.json')),
                         '--phase',phase,'--dtype',dt,'--backend',backend]
                if args.candidate:command+=['--candidate',str(args.candidate)]
                with (root/(phase+'.log')).open('w') as log:
                    code=subprocess.call(command,stdout=log,stderr=subprocess.STDOUT,timeout=95)
                assert code==0,(dt,backend,phase,code)
            passed(dt+'-'+backend+'-full-model-amp-and-fresh-resume')
        report['status']='passed';save();return

    if args.phase=='compile':
        from torch._inductor import metrics
        from torch._inductor.compile_fx import _warn_tf32_disabled
        import warnings
        assert not torch.cuda.is_tf32_supported()
        _warn_tf32_disabled.cache_clear()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');_warn_tf32_disabled()
        assert not caught,[str(x.message) for x in caught]
        passed('native-gfx1013-no-false-tf32-capability-warning')
        torch._dynamo.config.suppress_errors=False
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__();self.norm=torch.nn.LayerNorm(128)
                self.qkv=torch.nn.Linear(128,384);self.proj=torch.nn.Linear(128,128)
            def forward(self,x):
                b,n,d=x.shape
                q,k,v=self.qkv(self.norm(x)).reshape(b,n,3,4,32).permute(2,0,3,1,4).unbind(0)
                a=F.scaled_dot_product_attention(q,k,v,is_causal=True)
                return x+self.proj(a.transpose(1,2).reshape(b,n,d))
        for dt in [torch.float32,torch.float16,torch.bfloat16]:
            eager=Block().cuda();compiled_base=copy.deepcopy(eager)
            compiled=torch.compile(compiled_base,fullgraph=True,dynamic=True)
            before=metrics.generated_kernel_count
            for b,n in [(2,128),(3,96)]:
                x=torch.randn(b,n,128,device='cuda',requires_grad=True)
                y=x.detach().clone().requires_grad_()
                eager.zero_grad(set_to_none=True);compiled.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=dt,enabled=dt!=torch.float32),sdpa_kernel(SDPBackend.MATH):
                    a=eager(x);c=compiled(y)
                    a.square().mean().backward();c.square().mean().backward()
                errors=[]
                for av,cv in zip([a,x.grad]+[p.grad for p in eager.parameters()],
                                 [c,y.grad]+[p.grad for p in compiled_base.parameters()]):
                    relative=((av.float()-cv.float()).norm()/av.float().norm().clamp_min(1e-12)).item()
                    assert relative < (.0005 if dt==torch.float32 else .04),(dt,relative)
                    torch.testing.assert_close(av,cv,rtol=.002 if dt==torch.float32 else .06,
                                               atol=3e-5 if dt==torch.float32 else .005)
                    errors.append(relative)
                passed(f'compiled-eager-{dt}-batch{b}-length{n}',maximum_relative_l2=max(errors))
            assert metrics.generated_kernel_count>before
        report['status']='passed';save();return

    from transformers import Qwen3Config,Qwen3ForCausalLM
    cfg=Qwen3Config(vocab_size=257,hidden_size=128,intermediate_size=192,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=32,attention_dropout=0.)
    cfg._attn_implementation='sdpa'
    if args.backend=='tiled':
        from torch.nn.attention._gfx1013 import register_transformers_attention
        if args.candidate:
            spec=importlib.util.spec_from_file_location('bc250_amp_candidate',args.candidate)
            mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
            register_transformers_attention=mod.register_transformers_attention
        register_transformers_attention();cfg._attn_implementation='bc250_tiled'
    model=Qwen3ForCausalLM(cfg).cuda().train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    sched=torch.optim.lr_scheduler.StepLR(opt,step_size=1,gamma=.9)
    scaler=torch.amp.GradScaler('cuda',enabled=dtype==torch.float16,init_scale=32.)
    length=256 if dtype==torch.bfloat16 else 128
    root=args.output.parent;state_path=root/'checkpoint.pt';expected_path=root/'expected-next.pt'
    def step():
        # Exercise Python, CPU and GPU RNG restoration before an actual update.
        shift=random.randrange(8)
        ids=(torch.randint(0,249,(2,length))+shift).cuda()
        weight=.9+.1*torch.rand((),device='cuda')
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=dtype,enabled=dtype!=torch.float32),sdpa_kernel(SDPBackend.MATH):
            result=model(input_ids=ids,labels=ids,use_cache=False)
            loss=result.loss*weight
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())
        scale_before=scaler.get_scale()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        scaler.step(opt);scaler.update();sched.step()
        assert scaler.get_scale()>=scale_before,'Unexpected overflow/skipped update'
        return loss.item()
    if args.phase=='train-child':
        # All parameters train, and their initial gradients have a CPU float64 reference.
        cpu=copy.deepcopy(model).cpu().double()
        cpu.gradient_checkpointing_disable();cpu.set_attn_implementation('sdpa')
        ids=torch.arange(2*length).remainder(257).reshape(2,length)
        with torch.autocast('cuda',dtype=dtype,enabled=dtype!=torch.float32),sdpa_kernel(SDPBackend.MATH):
            result=model(input_ids=ids.cuda(),labels=ids.cuda(),use_cache=False)
            result.loss.backward()
        with sdpa_kernel(SDPBackend.MATH):
            ref=cpu(input_ids=ids,labels=ids,use_cache=False);ref.loss.backward()
        g=torch.cat([p.grad.float().cpu().flatten() for p in model.parameters()]).double()
        r=torch.cat([p.grad.flatten() for p in cpu.parameters()])
        rel=((g-r).norm()/r.norm()).item()
        assert rel < (.002 if dtype==torch.float32 else .04),rel
        passed('full-model-cpu-gradient-reference',relative_l2=rel,batch=2,length=length)
        del cpu,result,ref,g,r
        model.zero_grad(set_to_none=True)
        first=step()
        checkpoint=dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),
            scaler=scaler.state_dict(),rng=torch.get_rng_state(),gpu_rng=torch.cuda.get_rng_state(),
            python_rng=random.getstate())
        torch.save(checkpoint,state_path)
        second=step()
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),
                        scaler=scaler.state_dict(),loss=second,cpu_probe=torch.rand(8),
                        gpu_probe=torch.rand(8,device='cuda').cpu(),python_probe=random.random()),expected_path)
        passed('two-finite-full-model-updates',losses=[first,second],scale=scaler.get_scale())
    else:
        state=torch.load(state_path,map_location='cpu',weights_only=True)
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        sched.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
        torch.set_rng_state(state['rng']);torch.cuda.set_rng_state(state['gpu_rng']);random.setstate(state['python_rng'])
        loss=step()
        expected=torch.load(expected_path,map_location='cpu',weights_only=True)
        assert abs(loss-expected['loss'])<1e-6,(loss,expected['loss'])
        for name,p in model.state_dict().items():
            torch.testing.assert_close(p.cpu(),expected['model'][name],rtol=1e-6,atol=1e-7,msg=name)
        def compare_state(actual,reference):
            if isinstance(actual,torch.Tensor):
                torch.testing.assert_close(actual.cpu(),reference,rtol=1e-6,atol=1e-8)
            elif isinstance(actual,dict):
                assert actual.keys()==reference.keys()
                for name in actual:compare_state(actual[name],reference[name])
            elif isinstance(actual,(list,tuple)):
                assert len(actual)==len(reference)
                for a,b in zip(actual,reference):compare_state(a,b)
            else:assert actual==reference
        compare_state(opt.state_dict(),expected['optimizer'])
        assert sched.state_dict()==expected['scheduler'] and scaler.state_dict()==expected['scaler']
        torch.testing.assert_close(torch.rand(8),expected['cpu_probe'],rtol=0,atol=0)
        torch.testing.assert_close(torch.rand(8,device='cuda').cpu(),expected['gpu_probe'],rtol=0,atol=0)
        assert random.random()==expected['python_probe']
        passed('fresh-process-resume-next-update-and-all-rng-probes',loss=loss)
    report['status']='passed';save()


if __name__=='__main__':main()
