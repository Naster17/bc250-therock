#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Owned GPT2 fixture for the shared native FSDP reference/restart harness."""
import sys

import torch
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.pytorch_utils import Conv1D

from bc250_trainer_causal import CausalLabelCollator, causal_token_loss
from check_causal_d16 import contract_checks
from distributed_d16_trainer import OwnedData, cpu_token_ce
from distributed_d25_trainer import main

name = "GPT2 causal LM; tied embedding/head; right padding and shifted CE"
layer = "GPT2Block"
dataset = OwnedData
cpu_loss = cpu_token_ce
loss = causal_token_loss
collator = CausalLabelCollator
contracts = contract_checks
observed_modules = (torch.nn.Linear, Conv1D)
check_calls = 0


def model():
    config = GPT2Config(
        vocab_size=97,
        n_positions=32,
        n_ctx=32,
        n_embd=32,
        n_layer=2,
        n_head=4,
        n_inner=64,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_cache=False,
        attn_implementation="eager",
    )
    result = GPT2LMHeadModel(config)
    assert sum(p.numel() for p in result.parameters()) == 21280
    return result


def check(value):
    global check_calls
    weight = value.get_input_embeddings().weight
    assert weight is value.get_output_embeddings().weight
    aliases = {
        n.replace("_fsdp_wrapped_module.", "")
        for n, p in value.named_parameters(remove_duplicate=False)
        if p is weight
    }
    assert aliases == {"transformer.wte.weight", "lm_head.weight"}, aliases
    check_calls += 1


if __name__ == "__main__":
    if "--bert-control" in sys.argv:
        sys.argv.remove("--bert-control")
        main()  # Retained default of the shared D25 harness, in the new tools only.
    else:
        main(fixture=sys.modules[__name__])
