#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Owned frozen GPT2 with native low-rank Conv1D adapters; no PEFT package patch."""
import sys

import torch
from transformers.pytorch_utils import Conv1D

import distributed_d26_trainer as causal
from distributed_d25_trainer import main
from bc250_fsdp_frozen import BC250FrozenFSDPTrainer, frozen_fp32_policy
from check_fsdp_frozen_d27 import contracts as frozen_contracts

name = "frozen GPT2 with FP32 rank4 Conv1D adapters and tied frozen head"
frozen = True
layer = causal.layer
dataset = causal.dataset
cpu_loss = causal.cpu_loss
loss = causal.loss
collator = causal.collator


def contracts():
    return causal.contracts() + frozen_contracts()


observed_modules = causal.observed_modules
trainer_class = BC250FrozenFSDPTrainer
clip_limit = 0.005
policy = frozen_fp32_policy
check_calls = 0


class Adapter(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.down = torch.nn.Linear(base.weight.shape[0], 4, bias=False)
        self.up = torch.nn.Linear(4, base.weight.shape[1], bias=False)
        torch.nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.base(x) + self.up(self.down(x)) * 2.0


def model():
    result = causal.model()
    result.requires_grad_(False)
    modules = dict(result.named_modules())
    for name, value in list(modules.items()):
        if isinstance(value, Conv1D):
            parent, attribute = name.rsplit(".", 1)
            setattr(modules[parent], attribute, Adapter(value))
    assert sum(p.numel() for p in result.parameters()) == 24352
    assert sum(p.numel() for p in result.parameters() if p.requires_grad) == 3072
    return result


def check(value):
    global check_calls
    causal.check(value)
    assert not value.get_input_embeddings().weight.requires_grad
    check_calls += 1


def check_frozen(value, initial, *, full_gradient_views=False):
    current = {
        n.replace("_fsdp_wrapped_module.", ""): p
        for n, p in value.named_parameters()
        if not p.requires_grad
    }
    assert set(current) == set(initial)
    for n, p in current.items():
        if not full_gradient_views:
            assert p.grad is None, n
        assert torch.equal(p.detach().cpu(), initial[n]), n


if __name__ == "__main__":
    if "--retained-control" in sys.argv:
        sys.argv.remove("--retained-control")
        main(fixture=causal)
    else:
        main(fixture=sys.modules[__name__])
