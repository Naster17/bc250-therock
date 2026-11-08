#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CPU admission/gradient contracts for the explicit frozen FP32 policy."""
from types import SimpleNamespace
import torch
from bc250_trainer_fsdp import BC250FSDPTrainer
from bc250_fsdp_frozen import (
    BC250FrozenFSDPTrainer,
    FrozenFP32Policy,
    frozen_fp32_policy,
)


def contracts():
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    model[0].requires_grad_(False)
    policy = frozen_fp32_policy(model)
    owner = BC250FrozenFSDPTrainer.__new__(BC250FrozenFSDPTrainer)
    owner.model, owner.frozen_parameter_policy = model, policy
    owner._frozen_policy_sha = policy.sha256
    owner.args = SimpleNamespace(bf16=False, fp16=False, max_grad_norm=0.05)
    owner._validate_master_parameters()
    assert frozen_fp32_policy(model).sha256 == policy.sha256
    result = ["explicit-canonical-frozen-trainable-FP32-partition"]

    def refuse(fn):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError("Invalid frozen policy was admitted")

    for groups in [
        ([], ("b",)),
        ((), ("b",)),
        (("a", "a"), ("b",)),
        (("b", "a"), ("c",)),
        (("a",), ("a",)),
    ]:
        refuse(lambda: FrozenFP32Policy(*groups))
    result.append("empty-unsorted-duplicate-overlap-policy-refusals")
    legacy = BC250FSDPTrainer.__new__(BC250FSDPTrainer)
    legacy.model = model
    refuse(legacy._validate_master_parameters)
    result.append("ordinary-all-trainable-helper-still-refuses-frozen")
    for key in ["bf16", "fp16"]:
        setattr(owner.args, key, True)
        refuse(owner._validate_master_parameters)
        setattr(owner.args, key, False)
    owner.args.max_grad_norm = 0
    refuse(owner._validate_master_parameters)
    owner.args.max_grad_norm = 0.05
    result.append("AMP-and-disabled-clipping-refusals")
    model[0].weight.requires_grad_(True)
    owner.frozen_parameter_policy = frozen_fp32_policy(model)
    refuse(owner._validate_master_parameters)
    model[0].weight.requires_grad_(False)
    owner.frozen_parameter_policy = policy
    result.append("requires-grad-and-policy-drift-refusal")
    owner.optimizer = torch.optim.AdamW(
        p for p in model.parameters() if p.requires_grad
    )
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.ones_like(p)
    assert (
        owner._gradient_problem(list(model.parameters()), torch.device("cpu")).item()
        == 0
    )
    train = model[1].weight
    train.grad = None
    assert (
        owner._gradient_problem(list(model.parameters()), torch.device("cpu")).item()
        == 1
    )
    train.grad = torch.ones_like(train)
    model[0].weight.grad = torch.zeros_like(model[0].weight)
    assert (
        owner._gradient_problem(list(model.parameters()), torch.device("cpu")).item()
        == 1
    )
    model[0].weight.grad = None
    result.append("missing-trainable-and-present-frozen-gradient-refusals")
    owner.optimizer = torch.optim.AdamW(model.parameters())
    assert not owner._optimizer_matches()
    owner.optimizer = torch.optim.AdamW(
        p for p in model.parameters() if p.requires_grad
    )
    owner.optimizer.state[model[0].weight] = {}
    assert not owner._optimizer_matches()
    result.append("frozen-optimizer-membership-and-state-refusals")
    return result
