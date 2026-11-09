#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Actual PEFT BF16 working-parameter fixture with CPU all-adapter references."""
import sys
from dataclasses import replace
from types import SimpleNamespace

import torch
import distributed_d28_trainer as previous
from distributed_d25_trainer import main
from bc250_fsdp_peft_bf16 import BC250BF16PeftFSDPTrainer, validate_precision
from bc250_trainer_fsdp import fsdp_bf16_parameter_policy, fsdp_bf16_compute_policy

name = "PEFT GPT2 BF16 working params/FP32 frozen masters and reductions"
frozen = True
frozen_bf16 = True
peft = True
layer = previous.layer
dataset = previous.dataset
cpu_loss = previous.cpu_loss
loss = previous.loss
collator = previous.collator
observed_modules = previous.observed_modules
trainer_class = BC250BF16PeftFSDPTrainer
policy = previous.policy
policy_keyword = previous.policy_keyword
clip_limit = previous.clip_limit
wrap_units = previous.wrap_units
reduction_units = previous.reduction_units
model = previous.model
check_frozen = previous.check_frozen
export_audit = previous.export_audit
mutate = previous.mutate
check_calls = 0


def check(value):
    global check_calls
    previous.check(value)
    check_calls += 1


def contracts():
    result = previous.contracts()
    args = SimpleNamespace(
        bf16=True,
        fp16=False,
        gradient_checkpointing=False,
        fsdp_plugin_args=dict(mixed_precision_policy=fsdp_bf16_parameter_policy()),
    )
    validate_precision(args)
    for candidate in [
        None,
        fsdp_bf16_compute_policy(),
        replace(fsdp_bf16_parameter_policy(), reduce_dtype=torch.bfloat16),
        replace(fsdp_bf16_parameter_policy(), keep_low_precision_grads=True),
    ]:
        args.fsdp_plugin_args["mixed_precision_policy"] = candidate
        try:
            validate_precision(args)
        except ValueError:
            pass
        else:
            raise AssertionError("Unqualified frozen BF16 policy admitted")
    args.fsdp_plugin_args["mixed_precision_policy"] = fsdp_bf16_parameter_policy()
    for key, value in [
        ("bf16", False),
        ("fp16", True),
        ("gradient_checkpointing", True),
    ]:
        old = getattr(args, key)
        setattr(args, key, value)
        try:
            validate_precision(args)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "Unqualified precision/checkpoint combination admitted"
            )
        setattr(args, key, old)
    return result + [
        "explicit-frozen-BF16working-FP32master-reduce-policy",
        "missing-computeonly-lowreduce-lowgrad-FP16-checkpoint-refusals",
    ]


if __name__ == "__main__":
    if "--retained-control" in sys.argv:
        sys.argv.remove("--retained-control")
        main(fixture=previous)
    else:
        main(fixture=sys.modules[__name__])
