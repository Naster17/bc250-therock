#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Explicit pinned FP32 PEFT non-reentrant activation checkpointing admission."""
from functools import partial
import hashlib
from pathlib import Path

from torch.utils.checkpoint import checkpoint
from bc250_fsdp_peft import BC250PeftFSDPTrainer, unwrap

CHECKPOINT_KWARGS = dict(use_reentrant=False, preserve_rng_state=True)
CHECKPOINT_PINS = {
    "modeling_utils.py": "2f45c2139ce633e6e3eae9290675dd5815643b37c27f2cc4dd83586fb648f7bb",
    "modeling_layers.py": "9e4726c07cfc895d95f9a75b0e7d23b8561575080811695e70fa879b0a18dae4",
    "models/gpt2/modeling_gpt2.py": "97c3e08dadffb3e54a6c75d935af995b2cc9ba723fc49c16eb21056d49ee4ebd",
}


def validate_kwargs(args):
    if args.gradient_checkpointing is not True or (
        type(args.gradient_checkpointing_kwargs) is not dict
        or args.gradient_checkpointing_kwargs != CHECKPOINT_KWARGS
        or any(type(v) is not bool for v in args.gradient_checkpointing_kwargs.values())
    ):
        raise ValueError(
            "Explicit non-reentrant RNG-preserving checkpoint kwargs required"
        )


def checkpoint_modules(model):
    return [m for m in model.modules() if hasattr(m, "gradient_checkpointing")]


class BC250CheckpointPeftFSDPTrainer(BC250PeftFSDPTrainer):
    """Opt-in native Trainer checkpointing; currently qualified only for eager GPT2.

    FP32, named frozen policy, full checkpoint/export and root clipping guards
    are inherited. FSDP plugin checkpoint wrapping remains refused. Do not enable
    checkpointing on the model before constructing this Trainer.
    """

    def _validate_activation_checkpointing(self):
        import transformers
        from transformers import GPT2LMHeadModel

        validate_kwargs(self.args)
        root = Path(transformers.__file__).parent
        for name, sha in CHECKPOINT_PINS.items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != sha:
                raise RuntimeError(
                    "Checkpoint source requires requalification: " + name
                )
        base = unwrap(self.model).get_base_model()
        if (
            type(base) is not GPT2LMHeadModel
            or base.config.use_cache
            or base.config._attn_implementation != "eager"
            or base.is_gradient_checkpointing
            or not checkpoint_modules(base)
        ):
            raise ValueError(
                "Requires eager GPT2, cache disabled, checkpointing initially off"
            )
        self._checkpoint_started = False

    def _checkpoint_runtime_matches(self):
        try:
            validate_kwargs(self.args)
            modules = checkpoint_modules(self.model)
            return bool(modules) and all(
                m.gradient_checkpointing is True
                and isinstance(m._gradient_checkpointing_func, partial)
                and m._gradient_checkpointing_func.func is checkpoint
                and not m._gradient_checkpointing_func.args
                and m._gradient_checkpointing_func.keywords == CHECKPOINT_KWARGS
                for m in modules
            )
        except (ValueError, AttributeError, TypeError):
            return False

    def _partition_matches(self):
        return super()._partition_matches() and (
            not getattr(self, "_checkpoint_started", False)
            or self._checkpoint_runtime_matches()
        )

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Native Trainer enables checkpointing before wrapping/entering its loop.
        # The inherited collective clip and AdamW gates reject later live drift.
        self._checkpoint_started = True
        return super().training_step(model, inputs, num_items_in_batch)

    def _checkpoint_policy(self):
        result = super()._checkpoint_policy()
        result["activation_checkpointing"] = dict(CHECKPOINT_KWARGS)
        result["activation_checkpoint_sources"] = CHECKPOINT_PINS
        return result
