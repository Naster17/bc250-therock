#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Opt-in Trainer token statistics before prediction gathering (masked or causal)."""
import math

import torch


class CompactTokenMetrics:
    """Use preprocess as preprocess_logits_for_metrics and this object as compute_metrics.

    Each prediction row is [FP64 summed NLL, active targets, correct targets].
    Per-example rows preserve native gather_for_metrics tail de-duplication.
    Requires ordinary concatenated evaluation, not batch_eval_metrics/streaming.
    Labeled evaluation only: disable BOTH callbacks for full-logit/label-free
    prediction. Native labels and losses are still gathered. This bounds metric
    temporaries by chunk_tokens*vocab, not model logits or total dataset labels.
    """

    def __init__(self, *, causal=False, chunk_tokens=4096):
        if (
            type(causal) is not bool
            or type(chunk_tokens) is not int
            or chunk_tokens <= 0
        ):
            raise ValueError(
                "Expected boolean causal and positive integer chunk_tokens"
            )
        self.causal = causal
        self.chunk_tokens = chunk_tokens

    @torch.no_grad()
    def preprocess(self, logits, labels):
        if (
            not isinstance(logits, torch.Tensor)
            or not isinstance(labels, torch.Tensor)
            or logits.ndim != 3
            or labels.ndim != 2
            or logits.shape[:-1] != labels.shape
            or logits.device != labels.device
            or not logits.is_floating_point()
            or labels.dtype != torch.long
            or logits.shape[-1] < 1
        ):
            raise ValueError(
                "Expected one floating [N,L,V] tensor and same-device int64 [N,L] labels"
            )
        stats = torch.zeros(
            (labels.shape[0], 3), device=logits.device, dtype=torch.float64
        )
        # Row-wise slicing avoids flattening/copying the whole causal [N,L-1,V] view.
        for row in range(labels.shape[0]):
            z = logits[row, :-1] if self.causal else logits[row]
            y = labels[row, 1:] if self.causal else labels[row]
            indices = y.ne(-100).nonzero().flatten()
            stats[row, 1] = indices.numel()
            for start in range(0, indices.numel(), self.chunk_tokens):
                selected = indices[start : start + self.chunk_tokens]
                values = z[selected].double()
                targets = y[selected]
                if (
                    not torch.isfinite(values).all().item()
                    or targets.min().item() < 0
                    or targets.max().item() >= values.shape[-1]
                ):
                    raise ValueError(
                        "Nonfinite active logits or out-of-vocabulary labels"
                    )
                stats[row, 0] += torch.nn.functional.cross_entropy(
                    values, targets, reduction="sum"
                )
                stats[row, 2] += values.argmax(-1).eq(targets).sum()
        return stats

    def __call__(self, prediction):
        if isinstance(prediction.predictions, (tuple, list)):
            raise ValueError("Expected a single compact statistics array")
        stats = torch.as_tensor(prediction.predictions).detach().cpu()
        labels = torch.as_tensor(prediction.label_ids).detach().cpu()
        if (
            stats.ndim != 2
            or stats.shape[1] != 3
            or stats.dtype != torch.float64
            or labels.ndim != 2
            or labels.dtype != torch.long
            or stats.shape[0] != labels.shape[0]
            or not torch.isfinite(stats).all().item()
            or (stats < 0).any().item()
        ):
            raise ValueError(
                "Expected finite nonnegative FP64 [N,3] stats and int64 [N,L] labels"
            )
        targets = labels[:, 1:] if self.causal else labels
        counts = targets.ne(-100).sum(-1)
        if (
            not torch.equal(stats[:, 1], counts.double())
            or not torch.equal(stats[:, 2], stats[:, 2].round())
            or (stats[:, 2] > stats[:, 1]).any().item()
            or ((stats[:, 1] == 0) & (stats[:, 0] != 0)).any().item()
        ):
            raise ValueError(
                "Compact counts disagree with labels or invalid correct/NLL statistics"
            )
        count = int(counts.sum())
        if not count:
            raise ValueError("Token metrics require at least one supervised target")
        nll = float(stats[:, 0].sum()) / count
        try:
            perplexity = math.exp(nll)
        except OverflowError:
            perplexity = float("inf")
        return dict(
            token_nll=nll,
            token_perplexity=perplexity,
            token_accuracy=float(stats[:, 2].sum()) / count,
            active_tokens=count,
            evaluated_examples=stats.shape[0],
        )
