#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Optional collective fail-closed check for Trainer FP32/BF16 gradient updates."""
import torch
import torch.distributed as dist
from transformers import TrainerCallback


class NonFiniteGradientError(FloatingPointError):
    """Every live rank stops before its optimizer receives invalid gradients."""


class FiniteGradientCheck(TrainerCallback):
    """Stop FP32/BF16 jobs on all ranks; recover from a complete checkpoint.

    This deliberately does not silently skip a batch or patch Trainer internals.
    FP16 uses a different native scaler policy and is refused by this callback.
    Ranks must execute the callback together; peer death is not recovered here.
    """

    def on_train_begin(self, args, state, control, **kwargs):
        if args.fp16:
            raise ValueError(
                "FiniteGradientCheck supports FP32/BF16, not FP16 scaler policy"
            )

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        params = list(model.parameters())
        grads = [p.grad for p in params if p.grad is not None]
        bad = (
            (~torch.stack([torch.isfinite(g).all() for g in grads]).all()).to(
                torch.int32
            )
            if grads
            else torch.ones((), dtype=torch.int32, device=params[0].device)
        )
        if dist.is_initialized():
            dist.all_reduce(bad, op=dist.ReduceOp.MAX)
        if bad.item():
            raise NonFiniteGradientError(
                "Invalid gradients on a live rank; optimizer update refused on all ranks"
            )
