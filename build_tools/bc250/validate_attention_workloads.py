#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Long head128 references and bounded real Qwen3-1.7B LoRA comparisons."""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention._gfx1013 import register_transformers_attention
from torch.nn.attention._gfx1013 import scaled_dot_product_attention as tiled


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--phase', choices=['long', 'qwen'], required=True)
    ap.add_argument('--candidate', type=Path)
    ap.add_argument('--model', type=Path, default=Path('/home/nik/finetune-n7/models/Qwen3-1.7B-Base'))
    args = ap.parse_args()
    global tiled, register_transformers_attention
    if args.candidate:
        spec = importlib.util.spec_from_file_location('bc250_optimized_candidate', args.candidate)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        tiled = mod.scaled_dot_product_attention
        register_transformers_attention = mod.register_transformers_attention
    torch.set_num_threads(8)
    torch.manual_seed(716)
    report = dict(status='running', phase=args.phase, checks=[], measurements=[],
                  arch=torch.cuda.get_device_properties(0).gcnArchName)
    assert report['arch'].split(':')[0] == 'gfx1013'

    def save():
        args.output.write_text(json.dumps(report, indent=2)+'\n')

    def passed(name, **data):
        report['checks'].append(dict(name=name, **data))
        save()
        print('ATTENTION_WORKLOAD_PASS', report['checks'][-1], flush=True)

    if args.phase == 'long':
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            for h, hk, length in [(16,8,257),(16,8,513),(16,8,1024),(2,1,2048),(2,1,4096)]:
                if dtype == torch.float16 and length != 1024:
                    continue
                xs = [torch.randn(1, heads, length, 128, device='cuda', dtype=dtype,
                                  requires_grad=True) for heads in [h,hk,hk]]
                cpu = [x.detach().cpu().double().requires_grad_() for x in xs]
                gpu = [x.detach().clone().requires_grad_() for x in xs]
                mask = None
                causal = length != 513
                if not causal:
                    mask = torch.rand(length, length, device='cuda') > .2
                    mask[0, :] = False
                    mask[1, :256] = False
                opts = dict(is_causal=causal, enable_gqa=True, attn_mask=mask)
                grad = torch.randn_like(xs[0])
                out = tiled(*xs, **opts)
                out.backward(grad)
                with sdpa_kernel(SDPBackend.MATH):
                    ref = F.scaled_dot_product_attention(*cpu, **{**opts, 'attn_mask': None if mask is None else mask.cpu()})
                    ref.backward(grad.cpu().double())
                    gpu_out = F.scaled_dot_product_attention(*gpu, **opts)
                    gpu_out.backward(grad)
                relative, maximum = [], []
                tol = (8e-4, 3e-5) if dtype == torch.float32 else (.05, .03)
                bound = 1e-5 if dtype == torch.float32 else (.001 if dtype == torch.float16 else .006)
                for actual, expected, math_value in zip([out]+[x.grad for x in xs], [ref]+[x.grad for x in cpu],
                                                       [gpu_out]+[x.grad for x in gpu]):
                    a = actual.detach().cpu().double()
                    torch.testing.assert_close(a, expected.detach(), rtol=tol[0], atol=tol[1])
                    torch.testing.assert_close(actual, math_value, rtol=tol[0], atol=tol[1])
                    error = ((a-expected).norm()/expected.norm().clamp_min(1e-20)).item()
                    assert error < bound, error
                    relative.append(error)
                    maximum.append((a-expected).abs().max().item())
                passed(f'{dtype}-head128-h{h}-length{length}', relative_l2=relative,
                       max_absolute=maximum, causal=causal)
                del xs, cpu, gpu, grad, out, ref, gpu_out, actual, expected, math_value, a
                gc.collect()
        report['status'] = 'passed'
        save()
        return

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    from transformers.utils import logging as transformer_logging
    transformer_logging.disable_progress_bar()
    register_transformers_attention()
    report['model'] = str(args.model)
    report['config_sha256'] = hashlib.sha256((args.model/'config.json').read_bytes()).hexdigest()
    base = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
    report['base_parameters'] = sum(p.numel() for p in base.parameters())
    model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.,
        target_modules=['q_proj','v_proj'], task_type='CAUSAL_LM'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    model.train()
    report['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable = {n:p for n,p in model.named_parameters() if p.requires_grad}
    for length in [512, 1024]:
        # Synthetic tokens avoid touching private training data or saved adapters.
        ids = torch.randint(100, 10000, (1,length), device='cuda')
        refs = None
        for backend in ['sdpa', 'bc250_tiled']:
            model.base_model.model.set_attn_implementation(backend)
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            with sdpa_kernel(SDPBackend.MATH):
                result = model(input_ids=ids, labels=ids, use_cache=False)
                result.loss.backward()
            torch.cuda.synchronize()
            measurement = dict(length=length, backend=backend, loss=result.loss.item(),
                seconds=time.monotonic()-started, peak_allocated=torch.cuda.max_memory_allocated(),
                extra_peak_allocated=torch.cuda.max_memory_allocated()-baseline)
            gradients = {n:p.grad.detach().cpu().clone() for n,p in trainable.items()}
            sampled_logits = result.logits[:, ::64, :256].detach().float().cpu()
            if refs is None:
                refs = (measurement, gradients, sampled_logits)
            else:
                ref_measurement, ref_grads, ref_logits = refs
                assert abs(measurement['loss']-ref_measurement['loss']) < .025
                errors = {}
                for n, grad in gradients.items():
                    error = ((grad-ref_grads[n]).norm()/ref_grads[n].norm().clamp_min(1e-10)).item()
                    assert error < .06, (n,error)
                    errors[n] = error
                logit_error = ((sampled_logits-ref_logits).norm()/ref_logits.norm()).item()
                assert logit_error < .025, logit_error
                passed(f'qwen17-bf16-lora-length{length}', maximum_gradient_relative_l2=max(errors.values()),
                       sampled_logit_relative_l2=logit_error, loss_delta=measurement['loss']-ref_measurement['loss'])
            report['measurements'].append(measurement)
            save()
            del result, gradients, sampled_logits
            model.zero_grad(set_to_none=True)
            gc.collect()
    report['status'] = 'passed'
    save()


if __name__ == '__main__':
    main()
