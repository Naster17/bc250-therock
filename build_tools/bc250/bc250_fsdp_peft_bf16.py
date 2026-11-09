#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Explicit PEFT BF16 working parameters with FP32 frozen/master/reduce contracts."""
import hashlib
from pathlib import Path

import torch
from bc250_fsdp_peft import BC250PeftFSDPTrainer, unwrap
from bc250_fsdp_activation import CHECKPOINT_PINS
from bc250_trainer_fsdp import _bf16_parameter_policy_matches

PRECISION = dict(
    working_parameters="bfloat16",
    master_parameters="float32",
    reductions="float32",
    buffers="float32",
    keep_low_precision_grads=False,
    cast_forward_inputs=False,
    cast_root_forward_inputs=False,
    activation_checkpointing=False,
)


def validate_precision(args):
    if (
        args.bf16 is not True
        or args.fp16
        or args.gradient_checkpointing
        or not _bf16_parameter_policy_matches(
            (args.fsdp_plugin_args or {}).get("mixed_precision_policy")
        )
    ):
        raise ValueError(
            "Explicit BF16 working/FP32 master-reduce policy; checkpointing off required"
        )


class BC250BF16PeftFSDPTrainer(BC250PeftFSDPTrainer):
    """Pinned plain LoRA/eager GPT2 BF16 route; original helpers remain strict.

    Original parameters, including frozen weights, must be FP32 before wrapping.
    Only temporary FSDP working copies/allgathers use BF16. Do not cast the model
    or adapters to BF16 yourself. Native autocast has no gradient scaler here.
    """

    def _validate_frozen_precision(self):
        validate_precision(self.args)

    def _validate_activation_checkpointing(self):
        import transformers
        from transformers import GPT2LMHeadModel

        super()._validate_activation_checkpointing()
        root = Path(transformers.__file__).parent
        for name, wanted in CHECKPOINT_PINS.items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != wanted:
                raise RuntimeError(
                    "BF16 modeling source requires requalification: " + name
                )
        base = unwrap(self.model).get_base_model()
        if (
            type(base) is not GPT2LMHeadModel
            or base.config.use_cache
            or base.config._attn_implementation != "eager"
            or base.is_gradient_checkpointing
            or any(isinstance(m, torch.nn.Dropout) and m.p != 0 for m in base.modules())
        ):
            raise ValueError(
                "Requires eager GPT2/dropout0/cacheFalse/checkpointing off"
            )

    def _partition_matches(self):
        if not super()._partition_matches():
            return False
        try:
            validate_precision(self.args)
            if unwrap(self.model).get_base_model().is_gradient_checkpointing:
                return False
            acc = getattr(self, "accelerator", None)
            return acc is None or (
                acc.native_amp
                and acc.mixed_precision == "bf16"
                and acc.scaler is None
                and _bf16_parameter_policy_matches(
                    acc.state.fsdp_plugin.mixed_precision_policy
                )
                and not acc.state.fsdp_plugin.activation_checkpointing
            )
        except (ValueError, TypeError, AttributeError):
            return False

    def _checkpoint_policy(self):
        result = super()._checkpoint_policy()
        result["precision"] = dict(PRECISION)
        result["modeling_source_pins"] = CHECKPOINT_PINS
        return result
