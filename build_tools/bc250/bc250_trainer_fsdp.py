#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Opt-in source-pinned classic FSDP Trainer integration for two BC-250 hosts."""
import hashlib
import math
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


def _clip_limit(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("max_grad_norm must be a finite nonnegative number")
    return float(value)


def fsdp_bf16_compute_policy():
    """Explicit BF16 autocast compute with FP32 parameters, reductions and buffers.

    This does NOT enable BF16 parameter all-gathers or a low-precision storage
    policy. Pass in args.fsdp_plugin_args['mixed_precision_policy'] explicitly.
    """
    from torch.distributed.fsdp import MixedPrecision

    return MixedPrecision(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
        keep_low_precision_grads=False,
        cast_forward_inputs=False,
        cast_root_forward_inputs=False,
    )


def _bf16_compute_policy_matches(policy):
    from torch.distributed.fsdp import MixedPrecision

    return (
        isinstance(policy, MixedPrecision)
        and policy.param_dtype
        == policy.reduce_dtype
        == policy.buffer_dtype
        == torch.float32
        and not policy.keep_low_precision_grads
        and not policy.cast_forward_inputs
        and not policy.cast_root_forward_inputs
    )


def fsdp_bf16_parameter_policy():
    """BF16 working parameters/all-gathers; FP32 master shards and reductions."""
    from torch.distributed.fsdp import MixedPrecision

    return MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
        keep_low_precision_grads=False,
        cast_forward_inputs=False,
        cast_root_forward_inputs=False,
    )


def _bf16_parameter_policy_matches(policy):
    from torch.distributed.fsdp import MixedPrecision

    return (
        isinstance(policy, MixedPrecision)
        and policy.param_dtype == torch.bfloat16
        and policy.reduce_dtype == policy.buffer_dtype == torch.float32
        and not policy.keep_low_precision_grads
        and not policy.cast_forward_inputs
        and not policy.cast_root_forward_inputs
    )


def _qualified_bf16_policy(policy):
    return _bf16_compute_policy_matches(policy) or _bf16_parameter_policy_matches(
        policy
    )


class BC250FSDPTrainer(Trainer):
    """Separate classic FULL_SHARD/use_orig_params FP32 or explicit BF16-compute path; DDP helper unchanged.

    Native Trainer loop, FSDP wrapping/backward/optimizer/checkpoints retained.
    Explicit SHARDED_STATE_DICT; independent-host checkpoint payloads must be
    consolidated before restart. No FSDP2, FP16/scaler/offload/compile,
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
        limit = _clip_limit(args.max_grad_norm)
        if (
            args.fp16
            or args.gradient_checkpointing
            or args.torch_compile
            or args.deepspeed
        ):
            raise ValueError(
                "BC250FSDPTrainer requires FP32 or explicit BF16 compute; no FP16/checkpointing/compile/DeepSpeed"
            )
        if args.fsdp_plugin_args is None:
            raise ValueError("Explicit classic FSDP configuration is required")
        if args.bf16 and not _qualified_bf16_policy(
            args.fsdp_plugin_args.get("mixed_precision_policy")
        ):
            raise ValueError(
                "BF16 requires an explicit qualified compute or parameter policy"
            )
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
            or (not args.bf16 and plugin.mixed_precision_policy is not None)
            or (args.bf16 and not _qualified_bf16_policy(plugin.mixed_precision_policy))
            or plugin.activation_checkpointing
            or not plugin.sync_module_states
            or plugin.state_dict_type != StateDictType.SHARDED_STATE_DICT
        ):
            raise ValueError(
                "Requires two-rank classic FULL_SHARD/original FP32 params/no offload/SHARDED_STATE_DICT"
            )
        if args.bf16 and (
            not acc.native_amp
            or acc.mixed_precision != "bf16"
            or acc.scaler is not None
        ):
            raise ValueError("BF16 compute requires native autocast without a scaler")
        # Trainer already owns the full/short-window denominator and sync schedule.
        acc.gradient_accumulation_steps = 1
        if limit > 0:
            original_clip = acc.clip_grad_norm_

            def checked_clip(parameters, max_norm, norm_type=2):
                parameters = list(parameters)
                expected = list(self.model.parameters())
                if (
                    _clip_limit(max_norm) != limit
                    or norm_type != 2
                    or len(parameters) != len(expected)
                    or any(p is not q for p, q in zip(parameters, expected))
                ):
                    raise ValueError(
                        "Only the complete FSDP root L2 clipping call is qualified"
                    )
                bad = torch.zeros((), device=acc.device, dtype=torch.int32)
                for p in parameters:
                    if p.numel() == 0:
                        continue
                    g = p.grad
                    if (
                        g is None
                        or g.layout != torch.strided
                        or g.dtype != torch.float32
                        or g.device != acc.device
                    ):
                        bad.fill_(1)
                    else:
                        bad = torch.maximum(
                            bad, (~torch.isfinite(g)).any().to(torch.int32)
                        )
                torch.distributed.all_reduce(bad, op=torch.distributed.ReduceOp.MAX)
                if bad.item():
                    raise FloatingPointError(
                        "FSDP gradient refusal on a live rank; reload checkpoint"
                    )
                norm = original_clip(parameters, max_norm, norm_type=norm_type)
                bad = (~torch.isfinite(norm)).to(torch.int32)
                torch.distributed.all_reduce(bad, op=torch.distributed.ReduceOp.MAX)
                if bad.item():
                    raise FloatingPointError(
                        "FSDP norm refusal on a live rank; reload checkpoint"
                    )
                return norm

            # Per-instance public Accelerator method; no package/Trainer-loop fork.
            acc.clip_grad_norm_ = checked_clip

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
