#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Opt-in two-host dense DDP clipping and coordinated native FP16 overflow."""
from functools import partial
import math

import torch
import torch.distributed as dist
from transformers import TrainerCallback
from bc250_trainer import BC250Trainer
from bc250_trainer_finite import NonFiniteGradientError


def agree_bad(grads, device):
    unsupported = not grads or any(
        g.layout != torch.strided or g.device != device or g.dtype != torch.float32
        for g in grads
    )
    bad = (
        torch.ones((), dtype=torch.int32, device=device)
        if unsupported
        else (~torch.stack([torch.isfinite(g).all() for g in grads]).all()).to(
            torch.int32
        )
    )
    dist.all_reduce(bad, op=dist.ReduceOp.MAX)
    return bool(bad.item())


class _FinalGradientGate(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        owner = self.trainer
        if not owner._clip_ready:
            raise RuntimeError("Coordinated clipping did not run before optimizer")
        grads = [
            p.grad for p in model.parameters() if p.grad is not None and p.grad.numel()
        ]
        bad = agree_bad(grads, owner.accelerator.device)
        if bad and not owner._clip_known_overflow:
            raise NonFiniteGradientError(
                "Invalid gradient after clipping; all ranks must reload a complete checkpoint"
            )
        owner._clip_ready = False


class BC250ClippedTrainer(BC250Trainer):
    """Pinned D12 normalization plus a per-instance public clipping-method adapter.

    No global method patch, native scaler flag edit, or Trainer loop fork. Dense
    FP32 master gradients, one optimizer, two gfx1013 DDP ranks, L2 clipping only.
    User callbacks must not add/reorder callbacks or change gradients after the
    final gate. Late invalid gradients stop the job, rather than using a scaler
    decision that was cached before they appeared.
    """

    def create_accelerator_and_postprocess(self):
        super().create_accelerator_and_postprocess()
        acc = self.accelerator
        if acc.distributed_type.value != "MULTI_GPU" or acc.num_processes != 2:
            raise ValueError("BC250ClippedTrainer requires two-host DDP")
        self._clip_ready = False
        self._clip_known_overflow = False
        self._clip_gate = None
        # Decorate only this explicitly requested instance's public method.
        acc.clip_grad_norm_ = partial(
            self._coordinated_clip, native_clip=acc.clip_grad_norm_
        )

    def train(self, *args, **kwargs):
        bound = self.args.max_grad_norm
        if bound is None or not math.isfinite(bound) or bound <= 0:
            raise ValueError(
                "BC250ClippedTrainer requires finite positive max_grad_norm"
            )
        if self._clip_gate is not None:
            self.remove_callback(self._clip_gate)
        self._clip_gate = _FinalGradientGate(self)
        self.add_callback(self._clip_gate)  # After the caller's existing callbacks.
        return super().train(*args, **kwargs)

    def _coordinated_clip(self, parameters, max_norm, norm_type=2, *, native_clip):
        if norm_type != 2 or not math.isfinite(max_norm) or max_norm <= 0:
            raise ValueError("Only finite positive L2 clipping is supported")
        params = list(parameters)
        grads = [p.grad for p in params if p.grad is not None and p.grad.numel()]
        device = params[0].device
        invalid_format = torch.tensor(
            int(
                not grads
                or any(
                    g.layout != torch.strided
                    or g.dtype != torch.float32
                    or g.device != device
                    for g in grads
                )
            ),
            device=device,
            dtype=torch.int32,
        )
        dist.all_reduce(invalid_format, op=dist.ReduceOp.MAX)
        if invalid_format.item():
            raise ValueError(
                "Dense FP32 gradients on one device and one optimizer are required"
            )
        acc = self.accelerator
        states = [None] * acc.num_processes
        scaler = acc.scaler
        local_state = {} if scaler is None else scaler.state_dict()
        dist.all_gather_object(states, local_state)
        if any(value != states[0] for value in states) or (
            scaler is not None
            and (not math.isfinite(local_state["scale"]) or local_state["scale"] <= 0)
        ):
            raise NonFiniteGradientError(
                "Scaler state disagreement; update refused on all ranks"
            )
        known_overflow = agree_bad(grads, device)
        if known_overflow:
            if scaler is None:
                raise NonFiniteGradientError(
                    "Nonfinite BF16/FP32 gradient before clipping; update refused on all ranks"
                )
            # Native unscale must observe overflow on EVERY rank before caching its
            # decision. Public gradient mutation, no GradScaler private-state edit.
            with torch.no_grad():
                grads[0][(0,) * grads[0].ndim] = float("inf")
        norm = native_clip(params, max_norm, norm_type)
        bad_after = agree_bad(grads, device)
        if bad_after and not known_overflow:
            raise NonFiniteGradientError(
                "Clipping produced invalid gradients; reload a complete checkpoint on all ranks"
            )
        # Native norm reduction may overflow even when all input elements are finite.
        invalid_norm = (~torch.isfinite(norm)).to(torch.int32)
        dist.all_reduce(invalid_norm, op=dist.ReduceOp.MAX)
        if not known_overflow and invalid_norm.item():
            raise NonFiniteGradientError(
                "Nonfinite clipping norm; update refused on all ranks"
            )
        self._clip_known_overflow = known_overflow
        self._clip_ready = True
        return norm
