#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded scalar tiled SDPA references, checkpointing, and allocation evidence."""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import time
import warnings

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.candidate:
        spec = importlib.util.spec_from_file_location("bc250_sdpa_candidate", args.candidate)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        from torch.nn.attention import _gfx1013 as mod
    tiled = mod.scaled_dot_product_attention
    torch.set_num_threads(4)
    torch.manual_seed(714)
    report = {"status": "running", "torch": torch.__version__,
              "arch": torch.cuda.get_device_properties(0).gcnArchName,
              "cases": [], "rejections": [], "memory": [], "timings": []}
    assert report["arch"].split(":")[0] == "gfx1013"
    started = time.monotonic()

    def save():
        args.output.write_text(json.dumps(report, indent=2)+"\n")

    def reference(q, k, v, mask, causal, scale, gqa):
        if gqa:
            k = k.repeat_interleave(q.shape[1]//k.shape[1], 1)
            v = v.repeat_interleave(q.shape[1]//v.shape[1], 1)
        scores = (q @ k.transpose(-1, -2))*scale
        if causal:
            allowed = torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool).tril()
            scores = scores.masked_fill(~allowed, -float("inf"))
        if mask is not None:
            scores = scores.masked_fill(~mask, -float("inf")) if mask.dtype == torch.bool else scores+mask
        # Softmax(all -inf) is defined as zero by SDPA. Avoid NaNs in its derivative.
        empty = (scores == -float("inf")).all(-1, keepdim=True)
        probs = torch.softmax(scores.masked_fill(empty, 0), -1).masked_fill(empty, 0)
        return probs @ v

    configs = [(2, 4, 2, 17, 19, 33, "none", .17),
               (1, 2, 2, 16, 16, 32, "causal", .1767766952966369),
               (1, 2, 1, 3, 23, 64, "bool", .125),
               (1, 2, 1, 19, 7, 7, "additive", -.3),
               (1, 2, 1, 1, 31, 128, "none", .08838834764831845),
               (1, 2, 1, 23, 5, 16, "causal", .25)]
    if args.smoke:
        configs = configs[:1]
    dtypes = [torch.float32] if args.smoke else [torch.float32, torch.float16, torch.bfloat16]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for dtype in dtypes:
            for b, h, hk, m, n, d, mode, scale in configs:
                # Last-dimension slicing exercises packing and gradient view propagation.
                raw = [torch.randn(b, heads, length, d*2, device="cuda", dtype=dtype)
                       for heads, length in [(h, m), (hk, n), (hk, n)]]
                inputs = [x[..., ::2].detach().requires_grad_() for x in raw]
                cpu = [x.detach().cpu().double().requires_grad_() for x in inputs]
                gpu_ref = [x.detach().clone().requires_grad_() for x in inputs]
                mask = None
                if mode == "bool":
                    mask = torch.rand(b, 1, m, n, device="cuda") > .3
                    mask[:, :, 0, :] = False
                    mask[:, :, 1, :16] = False
                    mask[:, :, 1, 16:] = True
                elif mode == "additive":
                    mask = torch.randn(m, n, device="cuda")*.2
                    mask[0, :] = -float("inf")
                causal, gqa = mode == "causal", h != hk
                opts = dict(attn_mask=mask, is_causal=causal, scale=scale, enable_gqa=gqa)
                grad = torch.randn_like(inputs[0])
                out = tiled(*inputs, **opts)
                ref = reference(*cpu, None if mask is None else mask.cpu(), causal, scale, gqa)
                out.backward(grad)
                ref.backward(grad.cpu().double())
                with sdpa_kernel(SDPBackend.MATH):
                    math_out = F.scaled_dot_product_attention(*gpu_ref, **opts)
                    math_out.backward(grad)
                tol = {torch.float32: (5e-4, 3e-5), torch.float16: (8e-3, 3e-3),
                       torch.bfloat16: (5e-2, 3e-2)}[dtype]
                actuals = [out] + [x.grad for x in inputs]
                expected = [ref] + [x.grad for x in cpu]
                math_values = [math_out] + [x.grad for x in gpu_ref]
                errors = []
                relative_errors = []
                for actual, expected_cpu, expected_math in zip(actuals, expected, math_values):
                    torch.testing.assert_close(actual.cpu().double(), expected_cpu, rtol=tol[0], atol=tol[1])
                    torch.testing.assert_close(actual, expected_math, rtol=tol[0], atol=tol[1])
                    errors.append((actual.cpu().double()-expected_cpu).abs().max().item())
                    relative = ((actual.cpu().double()-expected_cpu).norm()/expected_cpu.norm().clamp_min(1e-20)).item()
                    assert relative < {torch.float32: 5e-6, torch.float16: 8e-4, torch.bfloat16: 6e-3}[dtype], relative
                    relative_errors.append(relative)
                report["cases"].append(dict(dtype=str(dtype), shape=[b,h,hk,m,n,d],
                                            mode=mode, scale=scale, max_abs_errors=errors,
                                            relative_l2_errors=relative_errors))
                save()
                print("TILED_SDPA_CASE_PASS", report["cases"][-1], flush=True)
        if not args.smoke:
            q = torch.randn(1, 2, 17, 32, device="cuda", requires_grad=True)
            kv = [torch.randn(1, 1, 19, 32, device="cuda", requires_grad=True) for _ in range(2)]
            a = checkpoint(lambda *x: tiled(*x, enable_gqa=True), q, *kv, use_reentrant=False)
            a.square().mean().backward()
            assert all(x.grad is not None and x.grad.isfinite().all() for x in [q, *kv])
            report["checkpoint_backward"] = "passed"
            for name, opts, exception in [
                ("dropout", {"dropout_p": .1}, NotImplementedError),
                ("mask_gradient", {"attn_mask": torch.zeros(17, 19, device="cuda", requires_grad=True)}, NotImplementedError),
                ("causal_mask", {"is_causal": True, "attn_mask": torch.ones(17, 19, device="cuda", dtype=torch.bool)}, ValueError),
                ("gqa_disabled", {}, ValueError),
                ("nonfinite_scale", {"scale": float("nan")}, ValueError),
            ]:
                try:
                    tiled(q, *kv, **({"enable_gqa": True} if name != "gqa_disabled" else {}), **opts)
                except exception:
                    report["rejections"].append(name)
                else:
                    raise AssertionError(name)
            # Empty key/query dimensions must return zeros and valid empty gradients.
            for m, n in [(3, 0), (0, 3)]:
                xs = [torch.randn(1, 1, length, 16, device="cuda", requires_grad=True) for length in (m,n,n)]
                y = tiled(*xs)
                y.sum().backward()
                assert all(torch.count_nonzero(x.grad).item() == 0 for x in xs)
                assert torch.count_nonzero(y).item() == 0
            report["empty_dimensions"] = "passed"
            # Keep this small enough for meaningful CPU-model references and no downloads.
            from transformers import LlamaConfig, LlamaForCausalLM
            mod.register_transformers_attention()
            cfg = LlamaConfig(vocab_size=127, hidden_size=64, intermediate_size=96,
                              num_hidden_layers=1, num_attention_heads=4,
                              num_key_value_heads=2, head_dim=16, attention_dropout=0.)
            cfg._attn_implementation = "bc250_tiled"
            model = LlamaForCausalLM(cfg).cuda().train()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            cfg_cpu = LlamaConfig(**cfg.to_dict())
            # CPU math SDPA defines fully masked rows as zero. Some eager model
            # softmax paths yield NaNs for these rows, so use the SDPA semantics
            # already independently checked by reference() above.
            cfg_cpu._attn_implementation = "sdpa"
            cpu_model = LlamaForCausalLM(cfg_cpu).double().train()
            cpu_model.load_state_dict(model.state_dict())
            ids = torch.arange(64).remainder(127).reshape(2, -1)
            padding = torch.ones_like(ids)
            padding[0, :3] = 0
            result = model(input_ids=ids.cuda(), attention_mask=padding.cuda(), labels=ids.cuda(), use_cache=False)
            result.loss.backward()
            with sdpa_kernel(SDPBackend.MATH):
                cpu_result = cpu_model(input_ids=ids, attention_mask=padding, labels=ids, use_cache=False)
                cpu_result.loss.backward()
            report["transformers_finite"] = {
                "gpu": bool(result.logits.isfinite().all()),
                "cpu": bool(cpu_result.logits.isfinite().all())}
            save()
            torch.testing.assert_close(result.logits.cpu().double(), cpu_result.logits, rtol=1e-3, atol=3e-5)
            for (name, p), (_, cp) in zip(model.named_parameters(), cpu_model.named_parameters()):
                if p.grad is not None:
                    torch.testing.assert_close(p.grad.cpu().double(), cp.grad, rtol=2e-3, atol=3e-5, msg=name)
            report["transformers_checkpoint_model"] = dict(loss=result.loss.item(), reference=cpu_result.loss.item())
            del model, cpu_model, result, cpu_result
            gc.collect()
            # Warm compiled kernels before peak/timing measurements. No CPU references
            # here: the tests above provide numerical evidence, these provide memory only.
            for length in [512, 1024, 2048]:
                xs = [torch.randn(1, 2, length, 32, device="cuda", requires_grad=True) for _ in range(3)]
                for name, fn in [("tiled", tiled), ("math", F.scaled_dot_product_attention)]:
                    with sdpa_kernel(SDPBackend.MATH):
                        fn(*xs).sum().backward()
                        for x in xs: x.grad = None
                        torch.cuda.synchronize()
                        baseline = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        fn(*xs).sum().backward()
                        end.record()
                        torch.cuda.synchronize()
                        report["memory"].append(dict(backend=name, length=length,
                            peak_extra_bytes=torch.cuda.max_memory_allocated()-baseline))
                        report["timings"].append(dict(backend=name, length=length,
                            forward_backward_ms=start.elapsed_time(end)))
                        for x in xs: x.grad = None
                del xs
            save()
        report["warnings"] = [str(x.message) for x in caught]
        save()
        assert not any("sdp_kernel" in str(x.message) or "not compiled with memory efficient" in str(x.message) for x in caught)
    torch.cuda.synchronize()
    report["seconds"] = time.monotonic()-started
    report["status"] = "passed"
    save()
    print("TILED_SDPA_PASS", len(report["cases"]), flush=True)


if __name__ == "__main__":
    main()
