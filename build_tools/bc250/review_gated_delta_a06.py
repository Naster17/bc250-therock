#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""A06 recheck: phase envelopes, operator counts and nonzero-adapter LoRA."""
import argparse
from collections import Counter
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import time

import torch
from torch.nn import bc250_gated_delta as ops
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils._python_dispatch import TorchDispatchMode

import test_bc250_gated_delta as fixtures
import validate_qwen35_training_model as existing


class CountOps(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.calls = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.calls[str(func)] += 1
        return func(*args, **(kwargs or {}))


def phased(xs: list[torch.Tensor], native: bool) -> tuple:
    events = []

    def mark():
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        events.append(e)

    mark()
    q, k, kb, vb, g = (
        ops.prepare_gated_delta(*xs, normalize=True)
        if native
        else fixtures.reference_prepare(*(x.float() for x in xs), True)
    )
    mark()
    mask = torch.ones(64, 64, device=q.device, dtype=torch.bool).triu(1)
    decay = (g.unsqueeze(-1) - g.unsqueeze(-2)).masked_fill(mask, -float("inf")).exp()
    system = (kb @ k.transpose(-1, -2)) * decay
    attention = (q @ k.transpose(-1, -2)) * decay
    ed = g.exp().unsqueeze(-1)
    rhs = kb * ed
    mark()
    u = torch.linalg.solve_triangular(system, vb, upper=False, unitriangular=True)
    w = torch.linalg.solve_triangular(system, rhs, upper=False, unitriangular=True)
    mark()
    q = q * ed
    k = k * (g[..., -1:] - g).exp().unsqueeze(-1)
    state = torch.zeros(
        xs[0].shape[0],
        xs[0].shape[2],
        xs[0].shape[-1],
        xs[2].shape[-1],
        device=q.device,
    )
    outputs = []
    for i in range(q.shape[2]):
        delta = u[:, :, i] - w[:, :, i] @ state
        outputs.append(q[:, :, i] @ state + attention[:, :, i] @ delta)
        state = (
            state * g[:, :, i, -1].exp()[..., None, None]
            + k[:, :, i].transpose(-1, -2) @ delta
        )
    y = (
        torch.stack(outputs, dim=2)
        .flatten(2, 3)[:, :, : xs[0].shape[1]]
        .transpose(1, 2)
        .to(xs[0].dtype)
        .contiguous()
    )
    mark()
    return y, events


def profile() -> list[dict]:
    xs = fixtures.inputs(1, 1024, 16, 128, 128, torch.bfloat16, "cuda")
    gy = torch.randn_like(xs[2]) * 0.1
    rows = []
    # Verify the instrumented composition before using its phase envelopes.
    validation = {}
    for native in (False, True):
        y, _ = phased(xs, native)
        fn = (
            ops.chunk_gated_delta_rule
            if native
            else existing.qwen.torch_chunk_gated_delta_rule
        )
        exact = fn(*xs, use_qk_l2norm_in_kernel=True)[0]
        torch.testing.assert_close(y, exact, rtol=0, atol=0)
        ref = torch.autograd.grad(exact, xs, gy)
        actual = torch.autograd.grad(y, xs, gy)
        # Slice-assignment versus stack changes BF16 accumulation ordering.
        # Reuse the existing complete-operator BF16 relative-L2 qualification.
        errors = [
            float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))
            for a, b in zip(actual, ref)
        ]
        assert max(errors) < 0.02
        validation[native] = errors
    samples = {False: [], True: []}
    for round_id in range(8):
        for native in ((False, True) if round_id % 2 == 0 else (True, False)):
            torch.cuda.synchronize()
            wall = time.perf_counter()
            y, events = phased(xs, native)
            torch.autograd.grad(y, xs, gy)
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            end.synchronize()
            if round_id >= 2:
                sample = dict(
                    zip(
                        ("prepare", "interactions", "solves", "scan_output"),
                        [events[i].elapsed_time(events[i + 1]) for i in range(4)],
                    )
                )
                sample.update(
                    backward=events[-1].elapsed_time(end),
                    total=events[0].elapsed_time(end),
                    wall_ms=(time.perf_counter() - wall) * 1000,
                )
                samples[native].append(sample)
    for native in (False, True):
        with CountOps() as counter:
            y, _ = phased(xs, native)
            torch.autograd.grad(y, xs, gy)
            torch.cuda.synchronize()
        values = samples[native]
        rows.append(
            dict(
                native=native,
                samples_ms=values,
                medians_ms={
                    k: statistics.median(x[k] for x in values) for k in values[0]
                },
                aten_calls=dict(counter.calls.most_common()),
                instrumented_vs_unmodified_gradient_relative_l2=validation[native],
                scope="GPU-event phase envelopes include host submission and event instrumentation; ATen counts are dispatcher calls, not kernel counts. Backward total includes all phases.",
            )
        )
    return rows


def models(rows: list[dict], quick: bool) -> None:
    original = existing.qwen.torch_chunk_gated_delta_rule

    def native(
        *args,
        cu_seqlens=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        **kwargs
    ):
        assert cu_seqlens is None and not use_cache
        return ops.chunk_gated_delta_rule(*args, **kwargs)

    try:
        for dtype, length, pairs in [
            (torch.float32, 65, 2),
            (torch.bfloat16, 65, 2),
            (torch.bfloat16, 1024, 5),
        ]:
            if quick:
                pairs = 1
            torch.manual_seed(61718)
            model = existing.make_model(True, dtype)
            # Synthetic nonzero adapters exercise gradients of both A and B.
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if "lora_B" in name:
                        p.normal_(0, 0.001)
            existing.ops.install(model)
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            ids = torch.randint(3, model.config.vocab_size, (1, length), device="cuda")

            def run(fn):
                existing.qwen.torch_chunk_gated_delta_rule = fn
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                with sdpa_kernel(SDPBackend.MATH):
                    loss = model(input_ids=ids, labels=ids, use_cache=False).loss
                    loss.backward()
                torch.cuda.synchronize()
                data = dict(
                    seconds=time.perf_counter() - start,
                    loss=float(loss.detach()),
                    peak_bytes=torch.cuda.max_memory_allocated(),
                )
                gradients = {
                    n: p.grad.detach().cpu().float().clone()
                    for n, p in model.named_parameters()
                    if p.requires_grad and p.grad is not None
                }
                assert all(bool(g.isfinite().all()) for g in gradients.values())
                return data, gradients

            ref_data, ref = run(original)
            cand_data, cand = run(native)
            assert ref.keys() == cand.keys() and len(ref) == 372
            metrics = {}
            for name, grad in ref.items():
                other = cand[name]
                metrics[name] = dict(
                    reference_norm=float(grad.norm()),
                    native_norm=float(other.norm()),
                    difference_norm=float((grad - other).norm()),
                    relative_l2=float(
                        (grad - other).norm() / grad.norm().clamp_min(1e-12)
                    ),
                )
            nonzero = sum(
                x["reference_norm"] > 0 and x["native_norm"] > 0
                for x in metrics.values()
            )
            maximum = max(x["relative_l2"] for x in metrics.values())
            global_error = (
                sum(x["difference_norm"] ** 2 for x in metrics.values())
                / max(sum(x["reference_norm"] ** 2 for x in metrics.values()), 1e-24)
            ) ** 0.5
            row = dict(
                dtype=str(dtype),
                length=length,
                gradients=metrics,
                nonzero_gradient_tensors=nonzero,
                max_relative_l2=maximum,
                global_relative_l2=global_error,
                initial_comparison=[ref_data, cand_data],
                samples=[],
            )
            rows.append(row)
            # Preserve prior pre-set limits; failures remain diagnostic evidence.
            row["numerics_passed"] = (
                nonzero == 372
                and maximum < (0.002 if dtype == torch.float32 else 0.04)
                and abs(ref_data["loss"] - cand_data["loss"]) < 0.015
            )
            del ref, cand
            for pair in range(pairs):
                order = [("reference", original), ("native", native)]
                if pair % 2:
                    order.reverse()
                sample = {}
                for name, fn in order:
                    data, gradients = run(fn)
                    del gradients
                    sample[name] = data
                row["samples"].append(sample)
            row["median_reference_seconds"] = statistics.median(
                x["reference"]["seconds"] for x in row["samples"]
            )
            row["median_native_seconds"] = statistics.median(
                x["native"]["seconds"] for x in row["samples"]
            )
            row["paired_ratios"] = [
                x["reference"]["seconds"] / x["native"]["seconds"]
                for x in row["samples"]
            ]
            print(
                dtype,
                length,
                "nonzero",
                nonzero,
                "max_relative_l2",
                maximum,
                "global_relative_l2",
                global_error,
                "paired_ratios",
                row["paired_ratios"],
                flush=True,
            )
            del model
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        existing.qwen.torch_chunk_gated_delta_rule = original


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--models", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--candidate", type=Path)
    args = ap.parse_args()
    global ops
    if args.candidate:
        spec = importlib.util.spec_from_file_location(
            "gated_delta_review_candidate", args.candidate
        )
        ops = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ops)
    torch.set_num_threads(4)
    torch.manual_seed(61718)
    result = dict(
        status="running",
        source_sha256=hashlib.sha256(Path(ops.__file__).read_bytes()).hexdigest(),
        module=str(ops.__file__),
        candidate=args.candidate is not None,
    )
    result["results"] = []
    try:
        if args.models:
            models(result["results"], args.quick)
        else:
            result["results"] = profile()
        result["status"] = (
            "passed"
            if not args.models or all(x["numerics_passed"] for x in result["results"])
            else "numerics_failed"
        )
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    assert result["status"] == "passed", "See preserved numerical diagnostics"


if __name__ == "__main__":
    main()
