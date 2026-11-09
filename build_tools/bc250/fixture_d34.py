#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fixed owned GPT2 fixture; shares D32's explicit CE and native AdamW recipe."""
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from capacity_plan_d34 import CONFIG
from distributed_d32_capacity import (
    loss as loss,
    optimizer as optimizer,
    named as named,
    full as full,
)


def build():
    torch.manual_seed(431)
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=CONFIG["vocab"],
            n_positions=CONFIG["positions"],
            n_ctx=CONFIG["positions"],
            n_embd=CONFIG["width"],
            n_layer=CONFIG["blocks"],
            n_head=CONFIG["heads"],
            n_inner=4 * CONFIG["width"],
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
            attn_implementation="eager",
            pad_token_id=0,
        )
    )


def data(index):
    values = torch.arange(CONFIG["global_batch"] * CONFIG["tokens"]).reshape(
        CONFIG["global_batch"], CONFIG["tokens"]
    )
    ids = ((values * 17 + index * 31) % (CONFIG["vocab"] - 1) + 1).long()
    return ids, ids.clone()


def gpu_backward(model, index, rank, world, traffic):
    ids, labels = data(index)
    local = CONFIG["global_batch"] // world
    denominator = CONFIG["global_batch"] * (CONFIG["tokens"] - 1)
    total = torch.zeros((), device="cuda")
    traffic["active"] = True
    try:
        for offset in range(local):
            # FSDP has one local micro: no full unsharded no_sync gradient.
            row = rank * local + offset
            value = (
                loss(
                    model(input_ids=ids[row : row + 1].cuda()).logits,
                    labels[row : row + 1].cuda(),
                    denominator,
                )
                * world
            )
            value.backward()
            total += value.detach()
    finally:
        traffic["active"] = False
    return total
