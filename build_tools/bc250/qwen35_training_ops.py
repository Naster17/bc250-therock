# SPDX-License-Identifier: MIT
"""BC-250 eager first-order Qwen3.5 training ops; frozen depthwise weights only.

Native scalar Triton convolution avoids MIOpen search for this bounded operation.
Chunked loss recomputes FP32 probabilities in backward; no full FP32 logits copy.
No fused gated-delta/FLA implementation, graph or higher-order gradient contract.
"""

import functools
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl


@functools.lru_cache(None)
def _native(device: torch.device) -> bool:
    return (
        torch.cuda.get_device_properties(device).gcnArchName.split(":")[0] == "gfx1013"
    )


@triton.jit
def _conv(
    X,
    W,
    B,
    Y,
    Z,
    L: tl.constexpr,
    C: tl.constexpr,
    N: tl.constexpr,
    XB: tl.constexpr,
    XC: tl.constexpr,
    XT: tl.constexpr,
    WC: tl.constexpr,
    WK: tl.constexpr,
    BS: tl.constexpr,
    K: tl.constexpr,
    BIAS: tl.constexpr,
    ACT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    t = i % L
    c = (i // L) % C
    b = i // (L * C)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for k in tl.static_range(K):
        source = t + k - (K - 1)
        x = tl.load(
            X + b * XB + c * XC + source * XT, (i < N) & (source >= 0), other=0
        ).to(tl.float32)
        w = tl.load(W + c * WC + k * WK, i < N, other=0).to(tl.float32)
        acc = acc + x * w
    if BIAS:
        acc = acc + tl.load(B + c * BS, i < N, other=0).to(tl.float32)
    # Match conv-output rounding before the separate activation in the reference.
    z = acc.to(Y.dtype.element_ty).to(tl.float32)
    if ACT:
        tl.store(Z + i, z, i < N)
        z = z * tl.sigmoid(z)
    tl.store(Y + i, z, i < N)


@triton.jit
def _conv_dx(
    G,
    W,
    Z,
    DX,
    L: tl.constexpr,
    C: tl.constexpr,
    N: tl.constexpr,
    WC: tl.constexpr,
    WK: tl.constexpr,
    K: tl.constexpr,
    ACT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    t = i % L
    c = (i // L) % C
    base = i - t
    acc = tl.full((BLOCK,), 0, tl.float32)
    for k in tl.static_range(K):
        target = t + (K - 1) - k
        valid = (i < N) & (target < L)
        g = tl.load(G + base + target, valid, other=0).to(tl.float32)
        if ACT:
            z = tl.load(Z + base + target, valid, other=0).to(tl.float32)
            sigmoid = tl.sigmoid(z)
            g = (
                (g * sigmoid * (1 + z * (1 - sigmoid)))
                .to(G.dtype.element_ty)
                .to(tl.float32)
            )
        w = tl.load(W + c * WC + k * WK, i < N, other=0).to(tl.float32)
        acc = acc + g * w
    tl.store(DX + i, acc, i < N)


class _FrozenCausalConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, activation):
        y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        z = torch.empty_like(y) if activation else y
        _conv[(triton.cdiv(x.numel(), 256),)](
            x,
            w,
            b if b is not None else w,
            y,
            z,
            x.shape[2],
            x.shape[1],
            x.numel(),
            *x.stride(),
            *w.stride(),
            b.stride(0) if b is not None else 0,
            w.shape[1],
            b is not None,
            activation,
            256,
            num_warps=4,
            enable_fp_fusion=False
        )
        ctx.save_for_backward(w, z)
        ctx.activation = activation
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        w, z = ctx.saved_tensors
        g = dy.contiguous()
        dx = torch.empty_like(g)
        _conv_dx[(triton.cdiv(g.numel(), 256),)](
            g,
            w,
            z,
            dx,
            g.shape[2],
            g.shape[1],
            g.numel(),
            *w.stride(),
            w.shape[1],
            ctx.activation,
            256,
            num_warps=4,
            enable_fp_fusion=False
        )
        return dx, None, None, None


def supports_causal_conv(
    x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None, activation: str | None
) -> bool:
    return (
        x.is_cuda
        and _native(x.device)
        and x.ndim == 3
        and w.ndim == 2
        and w.device == x.device
        and w.dtype == x.dtype
        and x.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and x.numel() > 0
        and x.shape[1] == w.shape[0]
        and 1 <= w.shape[1] <= 4
        and not w.requires_grad
        and (
            b is None
            or (
                b.device == x.device
                and b.dtype == x.dtype
                and b.shape == (x.shape[1],)
                and not b.requires_grad
            )
        )
        and activation in (None, "silu", "swish")
    )


def frozen_causal_conv(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    if not supports_causal_conv(x, w, b, activation):
        raise ValueError(
            "Native causal conv requires gfx1013 matching float tensors and frozen K<=4 depthwise weights/bias"
        )
    return _FrozenCausalConv.apply(x, w, b, activation is not None)


class _ChunkCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, denominator, chunk):
        flat = logits.view(-1, logits.shape[-1])
        target = targets.reshape(-1)
        loss = torch.zeros((), dtype=torch.float32, device=logits.device)
        for start in range(0, flat.shape[0], chunk):
            loss = loss + F.cross_entropy(
                flat[start : start + chunk].float(),
                target[start : start + chunk],
                ignore_index=-100,
                reduction="sum",
            )
        ctx.save_for_backward(logits, target, denominator)
        ctx.chunk = chunk
        return loss / denominator

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        logits, target, denominator = ctx.saved_tensors
        flat = logits.view(-1, logits.shape[-1])
        grad = torch.empty_like(logits)
        gf = grad.view_as(flat)
        for start in range(0, flat.shape[0], ctx.chunk):
            labels = target[start : start + ctx.chunk]
            valid = labels != -100
            probabilities = torch.softmax(
                flat[start : start + ctx.chunk].float(), dim=-1
            )
            rows = torch.arange(labels.numel(), device=logits.device)
            probabilities[rows, labels.masked_fill(~valid, 0)] -= valid.float()
            probabilities.mul_(dout.float() / denominator)
            probabilities.masked_fill_(~valid[:, None], 0)
            gf[start : start + ctx.chunk].copy_(probabilities)
        return grad, None, None, None


def chunked_causal_loss(
    logits,
    labels,
    vocab_size=None,
    num_items_in_batch=None,
    ignore_index=-100,
    shift_labels=None,
    chunk_tokens=64,
    **kwargs
):
    if not logits.is_cuda or not _native(logits.device) or not logits.is_contiguous():
        raise ValueError("Chunked loss requires contiguous gfx1013 logits")
    if (
        logits.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        or ignore_index != -100
    ):
        raise ValueError("Unsupported chunked loss dtype/ignore index")
    if vocab_size is not None and logits.shape[-1] != vocab_size:
        raise ValueError("Vocab size mismatch")
    if (
        not isinstance(chunk_tokens, int)
        or isinstance(chunk_tokens, bool)
        or not 1 <= chunk_tokens <= 256
    ):
        raise ValueError("Chunk size must be1..256")
    if shift_labels is None:
        shift_labels = F.pad(labels, (0, 1), value=ignore_index)[..., 1:].contiguous()
    target = shift_labels.to(logits.device)
    if target.shape != logits.shape[:-1] or target.dtype != torch.long:
        raise ValueError("Labels must match logit token shape and use int64")
    if num_items_in_batch is None:
        denominator = (target != -100).sum().float()
    else:
        denominator = torch.as_tensor(
            num_items_in_batch, device=logits.device, dtype=torch.float32
        )
        if (
            denominator.numel() != 1
            or denominator.requires_grad
            or not bool(denominator > 0)
        ):
            raise ValueError("Explicit item count must be a positive scalar")
    return _ChunkCE.apply(logits, target, denominator, chunk_tokens)


def install(model: torch.nn.Module) -> str:
    """Install only for the inspected Qwen3.5 text model on native gfx1013."""
    import transformers

    if transformers.__version__ != "5.19.0":
        raise ValueError("Qwen3.5 hooks are qualified only with Transformers 5.19.0")
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    parameter = next(base.parameters())
    if (
        getattr(base.config, "model_type", None) != "qwen3_5_text"
        or not parameter.is_cuda
        or not _native(parameter.device)
    ):
        raise ValueError("BC250 Qwen3.5 ops require the text-only gfx1013 model")
    import transformers.models.qwen3_5.modeling_qwen3_5 as module

    if not hasattr(module, "_bc250_original_causal_conv"):
        module._bc250_original_causal_conv = module.causal_conv1d_fn

        def causal(x, weight, bias=None, activation=None, **kwargs):
            if supports_causal_conv(x, weight, bias, activation):
                return frozen_causal_conv(x, weight, bias, activation)
            return module._bc250_original_causal_conv(
                x, weight, bias, activation, **kwargs
            )

        module.causal_conv1d_fn = causal
    if not hasattr(base, "_bc250_original_loss"):
        base._bc250_original_loss = base.loss_function
    base.loss_function = chunked_causal_loss
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(
        "[bc250] Qwen3.5 training ops "
        + digest
        + ": native frozen causal conv; chunk64 FP32 loss; gated-delta reference retained",
        flush=True,
    )
    return digest
