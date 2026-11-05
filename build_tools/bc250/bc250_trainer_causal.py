#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Explicit right-padded causal-label contract for the pinned Trainer helpers."""
import torch
from transformers import default_data_collator


class CausalLabelCollator:
    """Mask unpredicted position0 and padding before Trainer counts valid labels.

    Wrap a caller's collator or use the default stack collator for equal-shaped
    examples. Labels are supplied by the caller (including prompt ignore masks),
    cloned, never synthesized from inputs. Right-padded ordinary sequences only;
    no packed-document, left-padding or special position-ID policy is inferred.
    """

    def __init__(self, collator=default_data_collator):
        self.collator = collator

    def __call__(self, features):
        batch = self.collator(features)
        labels = batch.get("labels")
        inputs = batch.get("input_ids")
        if (
            not isinstance(inputs, torch.Tensor)
            or not isinstance(labels, torch.Tensor)
            or labels.dtype != torch.long
            or labels.ndim != 2
            or labels.shape != inputs.shape
            or labels.shape[1] < 2
        ):
            raise ValueError(
                "Causal labels must be int64 [batch,sequence>=2] matching input_ids"
            )
        labels = labels.clone()
        labels[:, 0] = -100
        mask = batch.get("attention_mask")
        if mask is not None:
            if (
                not isinstance(mask, torch.Tensor)
                or mask.shape != labels.shape
                or not ((mask == 0) | (mask == 1)).all().item()
                or (mask[:, 1:] > mask[:, :-1]).any().item()
                or (mask.sum(-1) < 2).any().item()
            ):
                raise ValueError(
                    "CausalLabelCollator requires binary right-padding masks with at least two real tokens"
                )
            labels.masked_fill_(mask == 0, -100)
        batch = dict(batch)
        batch["labels"] = labels
        return batch


def causal_token_loss(outputs, labels, num_items_in_batch=None):
    """Shifted token-summed CE divided by Trainer's GLOBAL window target count.

    Pair with CausalLabelCollator and average_tokens_across_devices=True. An
    all-ignored LOCAL micro yields zero when the GLOBAL count remains positive.
    A globally empty window or unspecified denominator is refused.
    """
    if num_items_in_batch is None or int(num_items_in_batch) <= 0:
        raise ValueError("Causal loss requires a positive global shifted-target count")
    logits = outputs.logits
    if (
        labels.ndim != 2
        or logits.shape[:-1] != labels.shape
        or labels[:, 0].ne(-100).any().item()
    ):
        raise ValueError(
            "Use full logits and CausalLabelCollator before Trainer target counting"
        )
    return (
        torch.nn.functional.cross_entropy(
            logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        / num_items_in_batch
    )
