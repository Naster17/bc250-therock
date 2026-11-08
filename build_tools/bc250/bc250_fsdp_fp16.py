#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Public-method FP16 clipping/scaler gates for the pinned classic FSDP Trainer."""
import math

import torch
import torch.distributed as dist


def agree_scaler(scaler):
    state = scaler.state_dict()
    states = [None] * 2
    dist.all_gather_object(states, state)
    if (
        not scaler.is_enabled()
        or any(s != states[0] for s in states)
        or not state
        or not math.isfinite(state["scale"])
        or state["scale"] <= 0
    ):
        raise FloatingPointError("FSDP scaler state disagreement; reload checkpoint")
    return state


def invalid_gradients(parameters, device, *, format_only=False):
    bad = torch.zeros((), device=device, dtype=torch.int32)
    for p in parameters:
        if not p.numel():
            continue
        g = p.grad
        if (
            g is None
            or g.layout != torch.strided
            or g.dtype != torch.float32
            or g.device != device
        ):
            bad.fill_(1)
        elif not format_only:
            bad = torch.maximum(bad, (~torch.isfinite(g)).any().to(torch.int32))
    dist.all_reduce(bad, op=dist.ReduceOp.MAX)
    return bool(bad.item())


def clip(owner, parameters, max_norm, norm_type):
    acc = owner.accelerator
    owner._fp16_clip_ready = False
    if invalid_gradients(parameters, acc.device, format_only=True):
        raise ValueError("FP16 FSDP requires present dense FP32 master gradients")
    owner._fp16_scaler_state = agree_scaler(acc.scaler)
    # This public call performs native shard-aware overflow synchronization.
    # Call the public root FSDP clip below directly to avoid a second unscale.
    acc.unscale_gradients()
    overflow = invalid_gradients(parameters, acc.device)
    owner._fp16_known_overflow = overflow
    if overflow:
        norm = torch.full((), float("inf"), device=acc.device)
    else:
        norm = owner.model.clip_grad_norm_(max_norm, norm_type=norm_type)
        bad = (~torch.isfinite(norm)).to(torch.int32)
        dist.all_reduce(bad, op=dist.ReduceOp.MAX)
        if bad.item() or invalid_gradients(parameters, acc.device):
            raise FloatingPointError("Invalid FSDP clipping result; reload checkpoint")
    owner._fp16_clip_ready = True
    return norm


def install_optimizer_gate(owner):
    optimizer = getattr(owner.optimizer, "optimizer", owner.optimizer)
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("Only native AdamW FP16 FSDP is qualified")
    if getattr(owner, "_fp16_gate_optimizer", None) is optimizer:
        return

    def before_step(raw, args, kwargs):
        # Invoked only when native ShardedGradScaler actually permits AdamW.
        state = agree_scaler(owner.accelerator.scaler)
        invalid = (
            not getattr(owner, "_fp16_clip_ready", False)
            or owner._fp16_known_overflow
            or state != owner._fp16_scaler_state
        )
        flag = torch.tensor(
            int(invalid), device=owner.accelerator.device, dtype=torch.int32
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item() or invalid_gradients(
            list(owner.model.parameters()), owner.accelerator.device
        ):
            raise FloatingPointError(
                "Invalid gradient/scaler after clipping; reload checkpoint"
            )
        owner._fp16_clip_ready = False
        owner._fp16_gate_calls = getattr(owner, "_fp16_gate_calls", 0) + 1

    optimizer.register_step_pre_hook(before_step)
    owner._fp16_gate_optimizer = optimizer
