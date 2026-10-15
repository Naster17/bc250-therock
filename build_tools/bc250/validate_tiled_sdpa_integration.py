#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Installed-path diagnostics, Qwen3 BF16 LoRA, cache decode and caller checks."""
import argparse
import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import warnings

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention._gfx1013 import scaled_dot_product_attention as tiled
from torch.nn.attention._gfx1013 import register_transformers_attention


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'status': 'running', 'checks': [], 'training': []}
    torch.set_num_threads(4)
    torch.manual_seed(715)

    def passed(name):
        torch.cuda.synchronize()
        report['checks'].append(name)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print('TILED_INTEGRATION_PASS', name, flush=True)

    q = torch.randn(1, 2, 13, 32, device='cuda', requires_grad=True)
    with warnings.catch_warnings(record=True) as quiet:
        warnings.simplefilter('always')
        with sdpa_kernel(SDPBackend.MATH):
            F.scaled_dot_product_attention(q, q, q).sum().backward()
        params = torch.backends.cuda.SDPAParams(q, q, q, None, 0., False, False)
        assert not torch.backends.cuda.can_use_efficient_attention(params, debug=False)
    assert not quiet, [str(x.message) for x in quiet]
    # The direct pybind diagnostic probe can use the c10 stderr handler rather
    # than Python's warning handler. Capture both channels for this one call.
    with tempfile.TemporaryFile() as diagnostic_output:
        saved_stderr = os.dup(2)
        try:
            os.dup2(diagnostic_output.fileno(), 2)
            with warnings.catch_warnings(record=True) as diagnostic:
                warnings.simplefilter('always')
                assert not torch.backends.cuda.can_use_efficient_attention(params, debug=True)
        finally:
            os.dup2(saved_stderr, 2)
            os.close(saved_stderr)
        diagnostic_output.seek(0)
        texts = [diagnostic_output.read().decode()] + [str(x.message) for x in diagnostic]
    assert any('not compiled with memory efficient attention' in x for x in texts)
    assert all('1Torch' not in x for x in texts)
    report['explicit_diagnostics'] = texts
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            try:
                F.scaled_dot_product_attention(q, q, q)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Built-in unavailable backend must not silently run')
    passed('quiet_normal_probe_and_retained_explicit_failure_diagnostics')
    # A caller with no backend context also gets a correct GPU math fallback.
    with warnings.catch_warnings(record=True) as quiet:
        warnings.simplefilter('always')
        out = F.scaled_dot_product_attention(q, q, q)
    assert not quiet
    with sdpa_kernel(SDPBackend.MATH):
        torch.testing.assert_close(out, F.scaled_dot_product_attention(q, q, q))
    passed('default_functional_sdpa_remains_quiet_math_fallback')
    # Extreme but finite logits exercise the online maximum across multiple tiles.
    for scale in [0., 10.]:
        xs = [torch.randn(1, 2, 35, 16, device='cuda', requires_grad=True) for _ in range(3)]
        refs = [x.detach().clone().requires_grad_() for x in xs]
        a = tiled(*xs, is_causal=True, scale=scale)
        with sdpa_kernel(SDPBackend.MATH):
            b = F.scaled_dot_product_attention(*refs, is_causal=True, scale=scale)
            b.sum().backward()
        a.sum().backward()
        for x, ref in zip([a]+[x.grad for x in xs], [b]+[x.grad for x in refs]):
            torch.testing.assert_close(x, ref, rtol=3e-3, atol=3e-4)
    passed('zero_scale_and_large_logits_output_and_gradients')
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        streamed = tiled(q, q, q)
    torch.cuda.current_stream().wait_stream(stream)
    with sdpa_kernel(SDPBackend.MATH):
        torch.testing.assert_close(streamed, F.scaled_dot_product_attention(q, q, q))
    passed('nondefault_gpu_stream')

    # Match the user's Qwen3-1.7B head geometry without loading its weights or
    # touching training data/checkpoints. Uneven sequence length spans five tiles.
    xs = [torch.randn(1, heads, 65, 128, device='cuda', dtype=torch.bfloat16,
                      requires_grad=True) for heads in [16, 8, 8]]
    cpu = [x.detach().cpu().double().requires_grad_() for x in xs]
    grad = torch.randn_like(xs[0])
    output = tiled(*xs, is_causal=True, enable_gqa=True)
    with sdpa_kernel(SDPBackend.MATH):
        ref = F.scaled_dot_product_attention(*cpu, is_causal=True, enable_gqa=True)
        ref.backward(grad.cpu().double())
    output.backward(grad)
    errors = []
    for actual, expected in zip([output]+[x.grad for x in xs], [ref]+[x.grad for x in cpu]):
        actual = actual.cpu().double()
        torch.testing.assert_close(actual, expected, rtol=.05, atol=.03)
        relative = ((actual-expected).norm()/expected.norm()).item()
        assert relative < .006, relative
        errors.append(relative)
    report['qwen17_head_geometry_relative_l2'] = errors
    passed('qwen17_bf16_head128_gqa16_to8_causal65_cpu_references')

    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM
    register_transformers_attention()
    config = Qwen3Config(vocab_size=127, hidden_size=64, intermediate_size=96,
                         num_hidden_layers=1, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=16, attention_dropout=0.)
    config._attn_implementation = 'bc250_tiled'
    base = Qwen3ForCausalLM(config).to(device='cuda', dtype=torch.bfloat16)
    model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4,
        target_modules=['q_proj','v_proj'], task_type='CAUSAL_LM', lora_dropout=0.))
    reference = copy.deepcopy(model).cpu().double()
    reference.base_model.model.set_attn_implementation('sdpa')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model.train()
    reference.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    cpu_opt = torch.optim.AdamW([p for p in reference.parameters() if p.requires_grad], lr=1e-3)
    ids = torch.arange(32).reshape(1, -1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for step in range(2):
            result = model(input_ids=ids.cuda(), labels=ids.cuda(), use_cache=False)
            result.loss.backward()
            with sdpa_kernel(SDPBackend.MATH):
                ref = reference(input_ids=ids, labels=ids, use_cache=False)
                ref.loss.backward()
            logit_error = ((result.logits.cpu().double()-ref.logits).norm()/ref.logits.norm()).item()
            assert logit_error < .025, logit_error
            torch.testing.assert_close(result.loss.cpu().double(), ref.loss.double(), rtol=.01, atol=.005)
            gradient_errors = {}
            for (name, p), (_, cp) in zip(model.named_parameters(), reference.named_parameters()):
                if p.requires_grad:
                    assert p.grad is not None and cp.grad is not None
                    error = ((p.grad.cpu().double()-cp.grad).norm()/cp.grad.norm().clamp_min(1e-12)).item()
                    assert error < .08, (name, error)
                    gradient_errors[name] = error
            report['training'].append(dict(step=step, loss=result.loss.item(),
                reference_loss=ref.loss.item(), logit_relative_l2=logit_error,
                gradient_relative_l2=gradient_errors))
            opt.step()
            cpu_opt.step()
            opt.zero_grad(set_to_none=True)
            cpu_opt.zero_grad(set_to_none=True)
        model.eval()
        # Cached decoding must attend all prior keys, rather than applying the
        # upper-left rectangular causal mask to a single newest query token.
        with torch.no_grad():
            prefill = model(input_ids=ids.cuda(), use_cache=True)
            decode = model(input_ids=torch.tensor([[5]], device='cuda'),
                           past_key_values=prefill.past_key_values, use_cache=True)
            full = model(input_ids=torch.cat([ids, torch.tensor([[5]])], -1).cuda(), use_cache=False)
            torch.testing.assert_close(decode.logits[:, -1], full.logits[:, -1], rtol=.04, atol=.006)
    report['warnings'] = [str(x.message) for x in caught]
    assert not any('sdp_kernel' in x or 'not compiled with memory efficient' in x for x in report['warnings'])
    passed('qwen3_bf16_lora_two_steps_checkpoint_gradients_and_cached_decode')
    for name in ['train_lora.py', 'chat.py']:
        path = Path('/home/nik/finetune-n7/scripts')/name
        tree = ast.parse(path.read_text())
        assert 'torch.backends.cuda.sdp_kernel' not in path.read_text()
        assert 'filterwarnings' not in path.read_text()
        if name == 'train_lora.py':
            guarded_backward = [node for node in ast.walk(tree) if isinstance(node, ast.With)
                and any(isinstance(item.context_expr, ast.Call) and
                        isinstance(item.context_expr.func, ast.Name) and
                        item.context_expr.func.id == 'math_ctx' for item in node.items)
                and any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                        and n.func.attr == 'backward' for n in ast.walk(node))]
            assert guarded_backward
        p = subprocess.run([sys.executable, str(path), '--help'], capture_output=True, text=True)
        assert p.returncode == 0 and '--attention-backend' in p.stdout, p.stderr
    passed('deployed_callers_new_api_backward_scope_and_cli_help')
    report['status'] = 'passed'
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print('TILED_SDPA_INTEGRATION_PASS', flush=True)


if __name__ == '__main__':
    main()
