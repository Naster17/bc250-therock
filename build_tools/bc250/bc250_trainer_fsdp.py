#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Opt-in source-pinned classic FSDP FP32 Trainer integration for two BC-250 hosts."""
import hashlib
from pathlib import Path

import torch
from transformers import Trainer

PINS = {
    "transformers/trainer.py": "fa7764933aaff94ae272cf3a545ed78e85c54c0b62ab1e309e3af6321095a095",
    "transformers/training_args.py": "33410b1c77482c37558a8b4fd6bf4ce76208919ecba1fbd8d05741e9dc041da4",
    "accelerate/accelerator.py": "8acd976354c667b858a61cff08baaaedd88dbef775ef4da3bb88c9327b3ddfd7",
    "accelerate/utils/fsdp_utils.py": "6b3e87c159515558a15795abbf8803d9dffddbe3be2fbca6444983ee6ba6628d",
    "accelerate/utils/dataclasses.py": "6f6db1018c1be71436e34baec4b48ef3add91be8056a884cb032576dd867e90d",
}


class BC250FSDPTrainer(Trainer):
    """Separate FP32 classic FULL_SHARD/use_orig_params path; DDP helper unchanged.

    Native Trainer loop, FSDP wrapping/backward/optimizer/checkpoints retained.
    Explicit SHARDED_STATE_DICT; independent-host checkpoint payloads must be
    consolidated before restart. No FSDP2, AMP/scaler/clipping/offload/compile,
    PEFT/frozen parameters or activation checkpointing qualification.
    """

    def create_accelerator_and_postprocess(self):
        import accelerate
        import transformers

        if (transformers.__version__, accelerate.__version__) != ("5.2.0", "1.12.0"):
            raise RuntimeError("BC250FSDPTrainer package pair requires requalification")
        for name, sha in PINS.items():
            package, relative = name.split("/", 1)
            root = (
                Path(transformers.__file__).parent
                if package == "transformers"
                else Path(accelerate.__file__).parent
            )
            if hashlib.sha256((root / relative).read_bytes()).hexdigest() != sha:
                raise RuntimeError(
                    "BC250FSDPTrainer source identity requires requalification: " + name
                )
        args = self.args
        if (
            args.bf16
            or args.fp16
            or args.max_grad_norm != 0
            or args.gradient_checkpointing
            or args.torch_compile
            or args.deepspeed
        ):
            raise ValueError(
                "BC250FSDPTrainer requires FP32/no clipping/checkpointing/compile/DeepSpeed"
            )
        if args.fsdp_plugin_args is None:
            raise ValueError("Explicit classic FSDP configuration is required")
        if any(
            not p.requires_grad or p.dtype != torch.float32
            for p in self.model.parameters()
        ):
            raise ValueError("Only trainable FP32 master parameters are qualified")
        super().create_accelerator_and_postprocess()
        acc = self.accelerator
        plugin = getattr(acc.state, "fsdp_plugin", None)
        from torch.distributed.fsdp import ShardingStrategy, StateDictType

        if (
            not self.is_fsdp_enabled
            or self.is_deepspeed_enabled
            or acc.num_processes != 2
            or plugin.fsdp_version != 1
            or not plugin.use_orig_params
            or plugin.sharding_strategy != ShardingStrategy.FULL_SHARD
            or plugin.cpu_offload.offload_params
            or plugin.mixed_precision_policy is not None
            or plugin.activation_checkpointing
            or not plugin.sync_module_states
            or plugin.state_dict_type != StateDictType.SHARDED_STATE_DICT
        ):
            raise ValueError(
                "Requires two-rank classic FULL_SHARD/original FP32 params/no offload/SHARDED_STATE_DICT"
            )
        # Trainer already owns the full/short-window denominator and sync schedule.
        acc.gradient_accumulation_steps = 1

    def _load_optimizer_and_scheduler(self, checkpoint):
        if checkpoint is None:
            return super()._load_optimizer_and_scheduler(checkpoint)
        # The pinned Accelerate generic DCP load uses the fresh empty optimizer
        # state as its template and silently leaves AdamW moments absent. Use
        # PyTorch's optimizer-aware reader to allocate them from DCP metadata.
        from torch.distributed.checkpoint import FileSystemReader
        from torch.distributed.checkpoint.optimizer import (
            load_sharded_optimizer_state_dict,
        )
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        root = Path(checkpoint)
        optimizer_root = root / "optimizer_0"
        if (
            not (optimizer_root / ".metadata").is_file()
            or not (root / "scheduler.pt").is_file()
        ):
            raise ValueError(
                "Complete sharded optimizer payload and scheduler are required"
            )
        raw = getattr(self.optimizer, "optimizer", self.optimizer)
        if not isinstance(raw, torch.optim.AdamW):
            raise ValueError("Only AdamW sharded restart is qualified")
        plugin = self.accelerator.state.fsdp_plugin
        with FSDP.state_dict_type(
            self.model,
            plugin.state_dict_type,
            plugin.state_dict_config,
            plugin.optim_state_dict_config,
        ):
            restored = load_sharded_optimizer_state_dict(
                model_state_dict=self.model.state_dict(),
                optimizer_key="optimizer",
                storage_reader=FileSystemReader(str(optimizer_root)),
            )
            local = FSDP.optim_state_dict_to_load(
                self.model, self.optimizer, restored["optimizer"]
            )
        self.optimizer.load_state_dict(local)
        self.lr_scheduler.load_state_dict(
            torch.load(root / "scheduler.pt", map_location="cpu", weights_only=True)
        )
        if any(
            p.numel() and not raw.state.get(p)
            for group in raw.param_groups
            for p in group["params"]
        ):
            raise RuntimeError(
                "Sharded AdamW restart left nonempty parameters without moments"
            )
