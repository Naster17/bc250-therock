#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""BF16 PEFT checkpointing fixture using retained all-gradient/restart references."""
import sys
from types import SimpleNamespace

import distributed_d30_trainer as previous
import distributed_d29_trainer as recompute
from distributed_d25_trainer import main
from bc250_fsdp_activation import CHECKPOINT_KWARGS, validate_kwargs, checkpoint_modules
from bc250_fsdp_peft_bf16 import validate_precision
from bc250_fsdp_peft_bf16_checkpoint import BC250BF16CheckpointPeftFSDPTrainer
from bc250_trainer_fsdp import fsdp_bf16_parameter_policy

name = "BF16 working-parameter PEFT GPT2 with native non-reentrant checkpointing"
frozen = True
frozen_bf16 = True
peft = True
layer = previous.layer
dataset = previous.dataset
cpu_loss = previous.cpu_loss
loss = previous.loss
collator = previous.collator
observed_modules = previous.observed_modules
trainer_class = BC250BF16CheckpointPeftFSDPTrainer
policy = previous.policy
policy_keyword = previous.policy_keyword
clip_limit = previous.clip_limit
wrap_units = previous.wrap_units
reduction_units = previous.reduction_units
model = previous.model
check_frozen = previous.check_frozen
export_audit = previous.export_audit
observe = recompute.observe
finalize = recompute.finalize
configure_args = recompute.configure_args
check_calls = 0


def check(value):
    global check_calls
    previous.check(value)
    check_calls += 1


def mutate(value):
    # Rank1-only post-clip drift: inherited AdamW gate must reject it before update.
    modules = checkpoint_modules(value)
    assert modules
    modules[0]._gradient_checkpointing_func.keywords["preserve_rng_state"] = False


def contracts():
    result = previous.contracts()
    args = SimpleNamespace(
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs=dict(CHECKPOINT_KWARGS),
        fsdp_plugin_args=dict(mixed_precision_policy=fsdp_bf16_parameter_policy()),
    )
    validate_precision(args, checkpointing=True)
    validate_kwargs(args)
    for bad in [
        None,
        {},
        dict(use_reentrant=True, preserve_rng_state=True),
        dict(use_reentrant=False, preserve_rng_state=False),
        dict(use_reentrant=0, preserve_rng_state=True),
    ]:
        args.gradient_checkpointing_kwargs = bad
        try:
            validate_kwargs(args)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid BF16 checkpoint kwargs admitted")
    args.gradient_checkpointing_kwargs = dict(CHECKPOINT_KWARGS)
    args.gradient_checkpointing = False
    try:
        validate_precision(args, checkpointing=True)
    except ValueError:
        pass
    else:
        raise AssertionError("Combined policy without checkpointing admitted")
    return result + [
        "explicit-BF16-plus-nonreentrant-RNG-checkpoint-policy",
        "missing-reentrant-RNGdisabled-nonbool-checkpoint-refusals",
    ]


if __name__ == "__main__":
    if "--retained-control" in sys.argv:
        sys.argv.remove("--retained-control")
        main(fixture=previous)
    else:
        main(fixture=sys.modules[__name__])
