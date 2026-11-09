#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Explicit combination of BF16 PEFT FSDP and native non-reentrant checkpointing."""
import torch
from bc250_fsdp_activation import BC250CheckpointPeftFSDPTrainer
from bc250_fsdp_peft import unwrap
from bc250_fsdp_peft_bf16 import (
    PRECISION,
    validate_precision,
    precision_runtime_matches,
)


class BC250BF16CheckpointPeftFSDPTrainer(BC250CheckpointPeftFSDPTrainer):
    """Pinned eager GPT2/dropout0, FP32 masters/reductions, BF16 working copies.

    Inherits native Trainer checkpoint enable/recompute, D28 full checkpoint and
    public adapter export, named frozen partition, root clipping and AdamW gates.
    Existing FP32 checkpoint and checkpointing-off BF16 helpers remain strict.
    """

    def _validate_frozen_precision(self):
        validate_precision(self.args, checkpointing=True)

    def _validate_activation_checkpointing(self):
        super()._validate_activation_checkpointing()
        base = unwrap(self.model).get_base_model()
        if any(isinstance(m, torch.nn.Dropout) and m.p != 0 for m in base.modules()):
            raise ValueError("BF16 checkpointing requires dropout0")

    def _partition_matches(self):
        return super()._partition_matches() and precision_runtime_matches(
            self, checkpointing=True
        )

    def _checkpoint_policy(self):
        result = super()._checkpoint_policy()
        result["precision"] = dict(PRECISION, activation_checkpointing=True)
        return result
