#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Isolate A06 BF16 model differences without changing installed sources."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import bc250_gated_delta as ops
from torch.nn.attention import sdpa_kernel, SDPBackend

import validate_qwen35_training_model as existing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    torch.manual_seed(61718)
    torch.set_num_threads(4)
    model = existing.make_model(True, torch.bfloat16)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(0, 0.001)
    existing.ops.install(model)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    ids = torch.randint(3, model.config.vocab_size, (1, 65), device="cuda")
    original = existing.qwen.torch_chunk_gated_delta_rule
    native_prepare = ops.prepare_gated_delta
    captured = []

    def aten_prepare(query, key, value, g, beta, normalize=True):
        # Mirror the pinned reference's operation ordering, including x*x.
        b, t, h, dk = query.shape
        query, key, value, beta, g = [
            x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
            for x in (query, key, value, beta, g)
        ]
        if normalize:
            query = existing.qwen.l2norm(query, dim=-1, eps=1e-6)
            key = existing.qwen.l2norm(key, dim=-1, eps=1e-6)
        query = query * dk**-0.5
        pad = (-t) % 64
        query, key, value = [
            torch.nn.functional.pad(x, (0, 0, 0, pad)) for x in (query, key, value)
        ]
        beta, g = [torch.nn.functional.pad(x, (0, pad)) for x in (beta, g)]
        vb, kb = value * beta.unsqueeze(-1), key * beta.unsqueeze(-1)
        query, key, kb, vb = [
            x.reshape(b, h, -1, 64, x.shape[-1]) for x in (query, key, kb, vb)
        ]
        return query, key, kb, vb, g.reshape(b, h, -1, 64).cumsum(-1)

    def mixed_prepare(*xs, normalize=True, forward_native):
        n = native_prepare(*xs, normalize=normalize)
        r = aten_prepare(*xs, normalize=normalize)
        # Separate test-only straight-through value and derivative substitutions.
        if forward_native:
            return tuple(a.detach() + (b - b.detach()) for a, b in zip(n, r))
        return tuple(b.detach() + (a - a.detach()) for a, b in zip(n, r))

    def wrapper(
        *xs,
        cu_seqlens=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        **kwargs
    ):
        assert cu_seqlens is None and not use_cache
        if not captured:
            captured.extend(x.detach().clone() for x in xs[:3])
            captured.extend(kwargs[n].detach().clone() for n in ["g", "beta"])
        return ops.chunk_gated_delta_rule(*xs, **kwargs)

    result = dict(
        status="running",
        source_sha256=hashlib.sha256(Path(ops.__file__).read_bytes()).hexdigest(),
        results=[],
    )
    baseline = None
    try:
        for label, fn, prepare in [
            ("stock", original, native_prepare),
            ("native", wrapper, native_prepare),
            ("ATen_preparation", wrapper, aten_prepare),
            (
                "ATen_forward_native_backward",
                wrapper,
                lambda *x, **kw: mixed_prepare(*x, **kw, forward_native=False),
            ),
            (
                "native_forward_ATen_backward",
                wrapper,
                lambda *x, **kw: mixed_prepare(*x, **kw, forward_native=True),
            ),
        ]:
            existing.qwen.torch_chunk_gated_delta_rule = fn
            ops.prepare_gated_delta = prepare
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            with sdpa_kernel(SDPBackend.MATH):
                loss = model(input_ids=ids, labels=ids, use_cache=False).loss
                loss.backward()
            torch.cuda.synchronize()
            grads = {
                n: p.grad.detach().cpu().float().clone()
                for n, p in model.named_parameters()
                if p.requires_grad and p.grad is not None
            }
            assert len(grads) == 372 and all(
                bool(g.isfinite().all()) for g in grads.values()
            )
            if baseline is None:
                baseline = grads
            metrics = {
                n: dict(
                    relative_l2=float(
                        (g - baseline[n]).norm() / baseline[n].norm().clamp_min(1e-12)
                    ),
                    difference_norm=float((g - baseline[n]).norm()),
                    reference_norm=float(baseline[n].norm()),
                )
                for n, g in grads.items()
            }
            worst = max(metrics, key=lambda n: metrics[n]["relative_l2"])
            row = dict(
                label=label,
                loss=float(loss.detach()),
                worst_parameter=worst,
                max_relative_l2=metrics[worst]["relative_l2"],
                global_relative_l2=(
                    sum(x["difference_norm"] ** 2 for x in metrics.values())
                    / sum(x["reference_norm"] ** 2 for x in metrics.values())
                )
                ** 0.5,
                gradients=metrics,
            )
            result["results"].append(row)
            print(
                label,
                row["loss"],
                row["max_relative_l2"],
                row["global_relative_l2"],
                worst,
                flush=True,
            )
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        with torch.no_grad():
            reference = aten_prepare(*captured)
            native = native_prepare(*captured)
            result["first_layer_preparation"] = [
                dict(
                    max_abs=float((a - b).abs().max()),
                    relative_l2=float((a - b).norm() / a.norm().clamp_min(1e-12)),
                    different_values=int((a != b).sum()),
                )
                for a, b in zip(reference, native)
            ]
        result["status"] = "passed"
    finally:
        ops.prepare_gated_delta = native_prepare
        existing.qwen.torch_chunk_gated_delta_rule = original
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
