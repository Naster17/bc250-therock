#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Explicit Trainer compatibility for the pinned Transformers5.2/Accelerate1.12 pair.

Trainer owns accumulation loss normalization and sync boundaries. Its Accelerator
must not divide the already-normalized loss again. This opt-in subclass keeps the
native Trainer loop, checkpoint handling and optimizer orchestration.
"""
import hashlib
from pathlib import Path

import torch
from transformers import Trainer


def masked_token_loss(outputs, labels, num_items_in_batch=None):
    """Masked-LM CE, summed over active tokens and normalized by Trainer's count.

    With average_tokens_across_devices=True Trainer supplies the global active
    token count for the entire accumulation window and compensates DDP averaging.
    This is NOT a causal-LM loss; labels are not shifted.
    """
    logits = outputs.logits
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="sum" if num_items_in_batch is not None else "mean",
    )
    if num_items_in_batch is not None:
        if int(num_items_in_batch) <= 0:
            raise ValueError(
                "The global accumulation window must contain an active label"
            )
        loss = loss / num_items_in_batch
    return loss


class BC250Trainer(Trainer):
    """Opt-in pinned-version integration; no global monkeypatch or package mutation."""

    def create_accelerator_and_postprocess(self):
        import accelerate
        import transformers

        if (transformers.__version__, accelerate.__version__) != ("5.2.0", "1.12.0"):
            raise RuntimeError(
                "BC250Trainer requires requalification for this package pair"
            )
        # Refuse a same-version source replacement as well as future versions:
        # upstream may repair this integration without changing our fixture API.
        for path, expected in [
            (
                Path(transformers.__file__).parent / "trainer.py",
                "fa7764933aaff94ae272cf3a545ed78e85c54c0b62ab1e309e3af6321095a095",
            ),
            (
                Path(accelerate.__file__).parent / "accelerator.py",
                "8acd976354c667b858a61cff08baaaedd88dbef775ef4da3bb88c9327b3ddfd7",
            ),
        ]:
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError(
                    "BC250Trainer source identity requires requalification"
                )
        super().create_accelerator_and_postprocess()
        if self.is_deepspeed_enabled or self.is_fsdp_enabled:
            raise ValueError("BC250Trainer currently qualifies plain/DDP only")
        # Native Trainer.training_step handles both mean-loss and token-count
        # normalization, including a shorter final accumulation window.
        self.accelerator.gradient_accumulation_steps = 1
