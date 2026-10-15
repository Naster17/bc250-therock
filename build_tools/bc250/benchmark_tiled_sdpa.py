#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Paired same-input measurements of original and candidate scalar attention."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics

import torch
from torch.nn.attention._gfx1013 import scaled_dot_product_attention as original


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--candidate', type=Path, required=True)
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location('bc250_optimized_benchmark', args.candidate)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    candidate = mod.scaled_dot_product_attention
    torch.manual_seed(717)
    torch.set_num_threads(4)
    report = dict(status='running', fixtures=[])
    for dtype in [torch.float32, torch.bfloat16]:
        for d, h, hk, length, causal in [(32,2,2,512,False), (128,16,8,257,True),
            (128,16,8,512,True), (128,16,8,1024,True), (128,2,1,2048,True),
            (128,4,2,513,False)]:
            xs = [torch.randn(1, heads, length, d, device='cuda', dtype=dtype,
                              requires_grad=True) for heads in [h,hk,hk]]
            grad = torch.randn_like(xs[0])
            opts = dict(is_causal=causal, enable_gqa=h!=hk)
            values = []
            for fn in [original, candidate]:
                y = fn(*xs, **opts)
                gs = torch.autograd.grad(y, xs, grad)
                values.append([y.detach(), *gs])
            for a,b in zip(*values):
                torch.testing.assert_close(a,b,rtol=.04 if dtype==torch.bfloat16 else .001,
                                           atol=.02 if dtype==torch.bfloat16 else 3e-5)
            del values,y,gs
            passes = []
            for label,fn in [('A1',original),('B1',candidate),('B2',candidate),('A2',original)]:
                for _ in range(2): torch.autograd.grad(fn(*xs, **opts), xs, grad)
                samples = []
                for _ in range(9):
                    start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record()
                    torch.autograd.grad(fn(*xs, **opts), xs, grad)
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end))
                passes.append(dict(label=label, median_ms=statistics.median(samples), samples_ms=samples))
            a = statistics.mean([passes[i]['median_ms'] for i in [0,3]])
            b = statistics.mean([passes[i]['median_ms'] for i in [1,2]])
            entry = dict(dtype=str(dtype),head_dim=d,heads=h,kv_heads=hk,length=length,causal=causal,
                         original_over_candidate=a/b, passes=passes,
                         original_drift=passes[3]['median_ms']/passes[0]['median_ms'])
            report['fixtures'].append(entry)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            print('ATTENTION_BENCH', {k:v for k,v in entry.items() if k!='passes'}, flush=True)
    report['status']='passed'
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
