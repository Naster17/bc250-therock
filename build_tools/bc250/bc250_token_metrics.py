#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Explicit dataset token-weighted metrics for native Trainer gathered predictions."""
import math

import torch


class TokenWeightedMetrics:
    """Stateless compute_metrics callback for FULL gathered logits and int64 labels.

    Use ordinary batch_eval_metrics=False/eval_do_concat_batches=True. Native
    Trainer gathering must remove repeated tail samples BEFORE this callback.
    Causal mode shifts exactly once; masked mode compares each position directly.
    Inputs already contain the whole evaluation set; chunking bounds temporary
    FP64 work, not native full-logit gathering/storage. No generation metrics.
    """

    def __init__(self, *, causal=False, chunk_tokens=4096):
        if not isinstance(chunk_tokens, int) or chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be positive")
        self.causal = causal
        self.chunk_tokens = chunk_tokens

    def __call__(self, prediction):
        if isinstance(prediction.predictions, (tuple, list)):
            raise ValueError("TokenWeightedMetrics requires a single full-logit array")
        logits = torch.as_tensor(prediction.predictions).detach().cpu()
        labels = torch.as_tensor(prediction.label_ids).detach().cpu()
        if (
            logits.ndim != 3
            or labels.ndim != 2
            or logits.shape[:-1] != labels.shape
            or labels.dtype != torch.long
            or not logits.is_floating_point()
        ):
            raise ValueError(
                "Expected full floating [examples,sequence,vocab] logits and int64 labels"
            )
        examples = labels.shape[0]
        if self.causal:
            logits = logits[:, :-1, :]
            labels = labels[:, 1:]
        indices = labels.reshape(-1).ne(-100).nonzero().flatten()
        count = indices.numel()
        if not count:
            raise ValueError("Token metrics require at least one supervised target")
        values = logits.reshape(-1, logits.shape[-1])
        targets = labels.reshape(-1)
        total = 0.0
        correct = 0
        for start in range(0, count, self.chunk_tokens):
            selected = indices[start : start + self.chunk_tokens]
            z = values[selected].double()
            y = targets[selected]
            if (
                not torch.isfinite(z).all().item()
                or y.min().item() < 0
                or y.max().item() >= z.shape[-1]
            ):
                raise ValueError("Nonfinite active logits or out-of-vocabulary labels")
            total += torch.nn.functional.cross_entropy(z, y, reduction="sum").item()
            correct += int(z.argmax(-1).eq(y).sum())
        nll = total / count
        try:
            perplexity = math.exp(nll)
        except OverflowError:
            perplexity = float("inf")
        return dict(
            token_nll=nll,
            token_perplexity=perplexity,
            token_accuracy=correct / count,
            active_tokens=count,
            evaluated_examples=examples,
        )
