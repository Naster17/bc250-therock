#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded reference gates for the application-scoped Qwen3.5 training helpers."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

import qwen35_training_ops as ops


def compare(actual, expected, dtype, label):
    a, b = actual.detach().cpu().double(), expected.detach().cpu().double()
    atol, rtol = {
        torch.float32: (4e-6, 3e-5),
        torch.float16: (0.004, 0.006),
        torch.bfloat16: (0.035, 0.035),
    }[dtype]
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol, msg=label)
    return float(
        torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-20)
    )


def operators():
    torch.manual_seed(481)
    torch.set_num_threads(4)
    rows = []
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for b, c, l, k in ((1, 3, 1, 1), (2, 7, 7, 2), (1, 32, 65, 4)):
            for activation in (None, "silu"):
                for layout in ("contiguous", "channel-last"):
                    x = torch.randn(b, c, l, device="cuda", dtype=dtype)
                    if layout == "channel-last":
                        x = x.transpose(1, 2).contiguous().transpose(1, 2)
                    x = x.detach().requires_grad_()
                    w = torch.randn(c, k, device="cuda", dtype=dtype) * 0.25
                    bias = (
                        torch.randn(c, device="cuda", dtype=dtype) * 0.1
                        if activation
                        else None
                    )
                    grad = torch.randn(b, c, l, device="cuda", dtype=dtype) * 0.25
                    y = ops.frozen_causal_conv(x, w, bias, activation)
                    (dx,) = torch.autograd.grad(y, x, grad)
                    ref = x.detach().cpu().double().requires_grad_()
                    z = F.conv1d(
                        ref,
                        w.cpu().double().unsqueeze(1),
                        None if bias is None else bias.cpu().double(),
                        padding=k - 1,
                        groups=c,
                    )[..., :l]
                    z = z.to(dtype).double()
                    expected = F.silu(z).to(dtype).double() if activation else z
                    (rdx,) = torch.autograd.grad(expected, ref, grad.cpu().double())
                    rows.append(
                        dict(
                            kind="conv",
                            dtype=str(dtype),
                            shape=[b, c, l],
                            k=k,
                            activation=activation,
                            layout=layout,
                            output_l2=compare(y, expected, dtype, "conv output"),
                            gradient_l2=compare(dx, rdx, dtype, "conv dx"),
                        )
                    )
        # Actual model's channel count and maximum configured training length.
        if dtype == torch.bfloat16:
            b, c, l, k = 1, 6144, 1024, 4
            x = (
                torch.randn(b, l, c, device="cuda", dtype=dtype)
                .transpose(1, 2)
                .detach()
                .requires_grad_()
            )
            w = torch.randn(c, k, device="cuda", dtype=dtype) * 0.1
            g = torch.randn_like(x) * 0.1
            y = ops.frozen_causal_conv(x, w, activation="silu")
            (dx,) = torch.autograd.grad(y, x, g)
            ref = x.detach().cpu().double().requires_grad_()
            z = (
                F.conv1d(ref, w.cpu().double().unsqueeze(1), padding=k - 1, groups=c)[
                    ..., :l
                ]
                .to(dtype)
                .double()
            )
            expected = F.silu(z).to(dtype).double()
            (rdx,) = torch.autograd.grad(expected, ref, g.cpu().double())
            rows.append(
                dict(
                    kind="conv-real-geometry",
                    shape=[b, c, l],
                    output_l2=compare(y, expected, dtype, "real conv"),
                    gradient_l2=compare(dx, rdx, dtype, "real conv dx"),
                )
            )
        for variant in ("mean", "count", "shift", "ignore"):
            logits = torch.randn(
                2, 13, 29, device="cuda", dtype=dtype, requires_grad=True
            )
            labels = torch.randint(29, (2, 13), device="cuda")
            labels[:, 3] = -100
            if variant == "ignore":
                labels.fill_(-100)
            target = (
                labels
                if variant == "shift"
                else F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
            )
            count = 37 if variant == "count" else None
            loss = ops.chunked_causal_loss(
                logits,
                labels,
                29,
                num_items_in_batch=count,
                shift_labels=target if variant == "shift" else None,
                chunk_tokens=7,
            )
            (grad,) = torch.autograd.grad(loss, logits)
            ref = logits.detach().cpu().double().requires_grad_()
            reference = F.cross_entropy(
                ref.reshape(-1, 29),
                target.cpu().reshape(-1),
                ignore_index=-100,
                reduction="mean" if count is None else "sum",
            )
            if count is not None:
                reference = reference / count
            (expected,) = torch.autograd.grad(reference, ref)
            if variant == "ignore":
                assert (
                    torch.isnan(loss)
                    and torch.isnan(reference)
                    and not bool(grad.any())
                )
            else:
                compare(loss, reference, torch.float32, "loss")
            rows.append(
                dict(
                    kind="loss",
                    dtype=str(dtype),
                    variant=variant,
                    gradient_l2=compare(grad, expected, dtype, "loss gradient"),
                )
            )
    # Trainable convolution weights must use the original implementation.
    x = torch.randn(1, 3, 5, device="cuda", requires_grad=True)
    w = torch.randn(3, 4, device="cuda", requires_grad=True)
    assert not ops.supports_causal_conv(x, w, None, "silu")
    try:
        ops.frozen_causal_conv(x, w, activation="silu")
    except ValueError:
        pass
    else:
        raise AssertionError("Trainable weights accepted")
    rows.append(dict(kind="trainable-weight-refusal"))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        storage = torch.randn(2, 7, 15, device="cuda")
        x = storage[..., 1:-1].detach().requires_grad_()
        w = torch.randn(7, 8, device="cuda")[:, ::2]
        bias = torch.randn(14, device="cuda")[::2]
        before = [t.clone() for t in (storage, w, bias)]
        y = ops.frozen_causal_conv(x, w, bias, "silu")
        (dx,) = torch.autograd.grad(y.sum(), x)
        ref = x.detach().cpu().double().requires_grad_()
        expected = F.silu(
            F.conv1d(
                ref,
                w.cpu().double().unsqueeze(1),
                bias.cpu().double(),
                padding=3,
                groups=7,
            )[..., :13]
            .float()
            .double()
        )
        (rdx,) = torch.autograd.grad(expected.sum(), ref)
        compare(y, expected, torch.float32, "stream output")
        compare(dx, rdx, torch.float32, "stream gradient")
        for t, saved in zip((storage, w, bias), before):
            assert torch.equal(t, saved), "Input/weight/bias storage was changed"
    torch.cuda.current_stream().wait_stream(stream)
    rows.append(dict(kind="nondefault-stream-padded-input-strided-weight-bias-guards"))
    return rows


def memory():
    torch.manual_seed(482)
    x = torch.randn(
        1, 1024, 248320, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    labels = torch.randint(x.shape[-1], x.shape[:-1], device="cuda")
    target = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous().flatten()
    rows = []
    baseline = None
    for mode in ("reference", "chunk64"):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_alloc = torch.cuda.memory_allocated()
        start = time.monotonic()
        if mode == "reference":
            loss = F.cross_entropy(x.float().view(-1, x.shape[-1]), target)
        else:
            loss = ops.chunked_causal_loss(x, labels)
        loss.backward()
        torch.cuda.synchronize()
        values = dict(
            kind="memory",
            mode=mode,
            loss=float(loss.detach()),
            seconds=time.monotonic() - start,
            baseline_allocated=start_alloc,
            peak_allocated=torch.cuda.max_memory_allocated(),
            peak_extra=torch.cuda.max_memory_allocated() - start_alloc,
        )
        # A fixed slice and total gradient norm check numerical agreement; small full gradients tested above.
        sample = x.grad.flatten()[::65537].clone()
        norm = torch.linalg.vector_norm(x.grad.float())
        if baseline is None:
            baseline = (values["loss"], sample, norm)
        else:
            assert abs(values["loss"] - baseline[0]) < 1e-5
            torch.testing.assert_close(sample, baseline[1], atol=1e-7, rtol=0.01)
            torch.testing.assert_close(norm, baseline[2], atol=1e-7, rtol=0.001)
        rows.append(values)
        x.grad = None
        del loss, sample, norm
    assert rows[1]["peak_extra"] < rows[0]["peak_extra"]
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--mode", choices=("operators", "memory"), default="operators")
    args = ap.parse_args()
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1013"
    result = dict(
        torch=torch.__version__,
        source_sha256=hashlib.sha256(Path(ops.__file__).read_bytes()).hexdigest(),
        mode=args.mode,
        checks=[],
    )
    try:
        result["checks"] = operators() if args.mode == "operators" else memory()
        result["passed"] = True
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
