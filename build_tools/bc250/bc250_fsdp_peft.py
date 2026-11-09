#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Pinned single-adapter PEFT LoRA FP32 admission with full native FSDP restart."""
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from bc250_fsdp_frozen import (
    BC250FrozenFSDPTrainer,
    FrozenFP32Policy,
    frozen_fp32_policy,
)

PEFT_PINS = {
    "peft/peft_model.py": "b847795c28bc4b00b3224c0dcea90edbee40ee60384cd104fdafd7056afcb13e",
    "peft/mapping_func.py": "714db3bc17ac5d3d72baea44a29f836199da731cecc7e96f9cf367ebcd6089a1",
    "peft/tuners/lora/model.py": "f619f6fccd6f2abc2ad51e4fde3f28a77a6b8104b7a94e1b0c18e341dafb48c7",
    "peft/tuners/lora/layer.py": "e8a47a49cf69f92ded68bf7ab286aeaf068f0ac809ab3465ad16f78a8f9f8b63",
    "peft/tuners/lora/config.py": "a33f9f057d7df7e352e796731c59c755197aa267d0d0f2698a516bb97ddc8997",
    "peft/tuners/tuners_utils.py": "99410963bc0bb2f194bef1ab01e65f3e24d5b066b90ea3242cd9edb11ead77a7",
    "peft/utils/save_and_load.py": "499d7bc36929d172da951b97ae81129834a5c69d6d4c434a1ee4359f256149ca",
    "peft/utils/other.py": "c5e3db7e853791235b110e5ecbf0a77095179b7ba42491da4bcb8c80a6c0c53d",
    "transformers/integrations/fsdp.py": "c9dd4f201b787782e56a857de895ab60439cd308af7321df163a3351cf90660c",
}


def package_identity():
    import peft
    import transformers

    if peft.__version__ != "0.18.1":
        raise RuntimeError("PEFT package requires requalification")
    roots = {
        "peft": Path(peft.__file__).parent,
        "transformers": Path(transformers.__file__).parent,
    }
    for name, wanted in PEFT_PINS.items():
        package, relative = name.split("/", 1)
        if (
            hashlib.sha256((roots[package] / relative).read_bytes()).hexdigest()
            != wanted
        ):
            raise RuntimeError("PEFT/FSDP source requires requalification: " + name)


def canonical(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, (set, frozenset)):
        return sorted(canonical(v) for v in value)
    if isinstance(value, (tuple, list)):
        return [canonical(v) for v in value]
    return value


def unwrap(model):
    return model.module if isinstance(model, FSDP) else model


def signature(model):
    from peft import PeftModelForCausalLM
    from peft.tuners.lora.layer import Linear
    from peft.tuners.lora import LoraConfig

    root = unwrap(model)
    if (
        type(root) is not PeftModelForCausalLM
        or set(root.peft_config) != {"default"}
        or root.active_adapters != ["default"]
    ):
        raise ValueError("One active default PEFT causal-LM LoRA adapter is required")
    cfg = root.peft_config["default"]
    if (
        type(cfg) is not LoraConfig
        or cfg.inference_mode
        or cfg.bias != "none"
        or cfg.lora_dropout != 0
        or cfg.use_dora
        or cfg.use_rslora
        or cfg.modules_to_save
        or cfg.rank_pattern
        or cfg.alpha_pattern
        or cfg.lora_bias
        or cfg.init_lora_weights is not True
        or type(cfg.r) is not int
        or cfg.r <= 0
        or isinstance(cfg.lora_alpha, bool)
        or not isinstance(cfg.lora_alpha, (int, float))
        or not math.isfinite(cfg.lora_alpha)
        or cfg.lora_alpha <= 0
    ):
        raise ValueError(
            "Only plain unmerged dropout0 LoRA without saved modules or variants is admitted"
        )
    layers = {}
    for name, module in root.named_modules():
        if isinstance(module, Linear):
            if (
                module.disable_adapters
                or module.merged_adapters
                or module.active_adapters != ["default"]
                or set(module.lora_A) != {"default"}
                or set(module.lora_B) != {"default"}
                or module.lora_variant
                or getattr(module, "ephemeral_gpu_offload", False)
                or module.r != {"default": cfg.r}
                or module.lora_alpha != {"default": cfg.lora_alpha}
                or module.scaling != {"default": cfg.lora_alpha / cfg.r}
            ):
                raise ValueError("LoRA layer activation/merge/offload state changed")
            layers[name.replace("_fsdp_wrapped_module.", "")] = dict(
                r=module.r,
                alpha=module.lora_alpha,
                scaling=module.scaling,
                use_dora=module.use_dora,
                lora_bias=module.lora_bias,
                fan_in_fan_out=module.fan_in_fan_out,
                dropout={
                    k: dict(type=type(v).__name__, p=getattr(v, "p", 0.0))
                    for k, v in module.lora_dropout.items()
                },
            )
    if not layers:
        raise ValueError("No admitted LoRA layers")
    base_config = root.get_base_model().config
    return json.dumps(
        canonical(
            dict(
                version="0.18.1",
                config=cfg.to_dict(),
                layers=layers,
                base_config=base_config.to_dict(),
                attention=getattr(base_config, "_attn_implementation", None),
            )
        ),
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class PeftFP32Policy:
    frozen: FrozenFP32Policy
    signature_json: str


def peft_fp32_policy(model):
    package_identity()
    result = PeftFP32Policy(frozen_fp32_policy(model), signature(model))
    if any(
        ".lora_A.default.weight" not in n and ".lora_B.default.weight" not in n
        for n in result.frozen.trainable_names
    ):
        raise ValueError("Only default LoRA A/B weights may train")
    return result


class BC250PeftFSDPTrainer(BC250FrozenFSDPTrainer):
    """Explicit pinned PEFT FP32 route; frozen/AMP/other-adapter restrictions remain."""

    def __init__(self, *args, peft_parameter_policy, **kwargs):
        package_identity()
        if type(peft_parameter_policy) is not PeftFP32Policy:
            raise ValueError("Explicit peft_fp32_policy required")
        self.peft_parameter_policy = peft_parameter_policy
        super().__init__(
            *args, frozen_parameter_policy=peft_parameter_policy.frozen, **kwargs
        )

    def _partition_matches(self):
        if not super()._partition_matches():
            return False
        try:
            return signature(self.model) == self.peft_parameter_policy.signature_json
        except (ValueError, TypeError, AttributeError):
            return False

    def _checkpoint_policy(self):
        result = super()._checkpoint_policy()
        result["peft_signature"] = json.loads(self.peft_parameter_policy.signature_json)
        result["peft_source_pins"] = PEFT_PINS
        return result

    def _save_optimizer_and_scheduler(self, output_dir):
        # Pinned Trainer requests adapter_only=True. This route intentionally
        # saves the full frozen base too, so restart needs no external base reload.
        from accelerate.utils import save_fsdp_model, save_fsdp_optimizer

        save_fsdp_model(
            self.accelerator.state.fsdp_plugin,
            self.accelerator,
            self.model,
            output_dir,
            adapter_only=False,
        )
        save_fsdp_optimizer(
            self.accelerator.state.fsdp_plugin,
            self.accelerator,
            self.optimizer,
            self.model,
            output_dir,
        )
        if self.args.should_save:
            torch.save(
                self.lr_scheduler.state_dict(), Path(output_dir) / "scheduler.pt"
            )

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        from accelerate.utils import load_fsdp_model

        load_fsdp_model(
            self.accelerator.state.fsdp_plugin,
            self.accelerator,
            model if model is not None else self.model,
            resume_from_checkpoint,
            adapter_only=False,
        )

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        # Separate ordinary PEFT artifact; it is not the full training checkpoint.
        folder = (
            Path(self.args.output_dir)
            / f"checkpoint-{self.state.global_step}"
            / "peft-adapter"
        )
        with FSDP.summon_full_params(self.model, writeback=False):
            root = unwrap(self.model)
            state = {
                n.replace("_fsdp_wrapped_module.", ""): p.detach().cpu().clone()
                for n, p in root.named_parameters()
                if p.requires_grad
            }
        if self.args.should_save:
            root.save_pretrained(folder, state_dict=state, save_embedding_layers=False)
