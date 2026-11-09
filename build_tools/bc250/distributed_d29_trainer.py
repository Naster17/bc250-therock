#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Pinned PEFT checkpointing fixture with retained CPU/restart reference harness."""
import sys
from types import SimpleNamespace

import torch
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
import distributed_d28_trainer as previous
from distributed_d25_trainer import main
from bc250_fsdp_activation import (
    BC250CheckpointPeftFSDPTrainer,
    CHECKPOINT_KWARGS,
    validate_kwargs,
)

name = "FP32 PEFT GPT2 native non-reentrant activation checkpointing"
frozen = True
layer = previous.layer
dataset = previous.dataset
cpu_loss = previous.cpu_loss
loss = previous.loss
collator = previous.collator
observed_modules = previous.observed_modules
trainer_class = BC250CheckpointPeftFSDPTrainer
policy = previous.policy
policy_keyword = previous.policy_keyword
clip_limit = previous.clip_limit
wrap_units = previous.wrap_units
reduction_units = previous.reduction_units
check_frozen = previous.check_frozen
export_audit = previous.export_audit
mutate = previous.mutate
check_calls = 0
# The shared harness uses this flag to bind the actual PEFT helper provenance.
peft = True
counts = []


def model():
    return previous.model()


def check(value):
    global check_calls
    previous.check(value)
    check_calls += 1


def contracts():
    result = previous.contracts()
    validate_kwargs(
        SimpleNamespace(
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs=dict(CHECKPOINT_KWARGS),
        )
    )
    for bad in [
        None,
        {},
        dict(use_reentrant=True),
        dict(use_reentrant=False),
        dict(use_reentrant=False, preserve_rng_state=False),
        dict(use_reentrant=0, preserve_rng_state=True),
    ]:
        try:
            validate_kwargs(
                SimpleNamespace(
                    gradient_checkpointing=True, gradient_checkpointing_kwargs=bad
                )
            )
        except ValueError:
            pass
        else:
            raise AssertionError("Unsupported checkpoint kwargs admitted")
    return result + [
        "explicit-nonreentrant-RNG-checkpoint-policy",
        "missing-reentrant-RNGdisabled-nonbool-refusals",
    ]


def configure_args(args):
    args.gradient_checkpointing = True
    args.gradient_checkpointing_kwargs = dict(CHECKPOINT_KWARGS)


def observe(value, result):
    for module in value.modules():
        if isinstance(module, GPT2Block):
            counter = [0]
            counts.append(counter)

            def hook(module, inputs, counter=counter):
                if module.training and inputs[0].is_cuda:
                    counter[0] += 1

            module.register_forward_pre_hook(hook)
    assert len(counts) == 2
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    result["allocator_initial_bytes"] = torch.cuda.memory_allocated()


def finalize(result):
    # Seven actual seed micros, three replay micros. Each block runs forward
    # and actual backward recomputation; CPU oracle hooks are absent.
    expected = 14 if len(result["transcript"]) == 4 else 6
    assert [c[0] for c in counts] == [expected, expected], counts
    result["checkpoint_block_calls"] = [c[0] for c in counts]
    result["allocator_peak_bytes"] = torch.cuda.max_memory_allocated()
    result["allocator_reserved_peak_bytes"] = torch.cuda.max_memory_reserved()
    result["checks"].append("GPU-both-blocks-forward-and-backward-recomputation")


if __name__ == "__main__":
    if "--retained-control" in sys.argv:
        sys.argv.remove("--retained-control")
        main(fixture=previous)
    else:
        main(fixture=sys.modules[__name__])
