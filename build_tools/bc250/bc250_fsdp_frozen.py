#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Explicit FP32 frozen/trainable admission; ordinary FSDP Trainer stays strict."""
from dataclasses import dataclass
import hashlib
import json

import torch
import torch.distributed as dist

from bc250_trainer_fsdp import BC250FSDPTrainer, _clip_limit


def named(model):
    return {
        n.replace("_fsdp_wrapped_module.", ""): p for n, p in model.named_parameters()
    }


@dataclass(frozen=True)
class FrozenFP32Policy:
    trainable_names: tuple[str, ...]
    frozen_names: tuple[str, ...]

    def __post_init__(self):
        for group in [self.trainable_names, self.frozen_names]:
            if (
                type(group) is not tuple
                or not group
                or any(type(n) is not str or not n for n in group)
                or tuple(sorted(set(group))) != group
            ):
                raise ValueError(
                    "Policy requires nonempty sorted unique parameter names"
                )
        if set(self.trainable_names) & set(self.frozen_names):
            raise ValueError("Frozen and trainable names must be disjoint")

    @property
    def sha256(self):
        return hashlib.sha256(
            json.dumps(
                [self.trainable_names, self.frozen_names], separators=(",", ":")
            ).encode()
        ).hexdigest()


def frozen_fp32_policy(model):
    """Capture the caller's explicitly configured requires_grad partition."""
    return FrozenFP32Policy(
        tuple(sorted(n for n, p in model.named_parameters() if p.requires_grad)),
        tuple(sorted(n for n, p in model.named_parameters() if not p.requires_grad)),
    )


class BC250FrozenFSDPTrainer(BC250FSDPTrainer):
    """Opt-in FP32 mixed frozen/trainable FULL_SHARD with exact named policy.

    No PEFT-package, AMP, offload or activation-checkpointing admission is implied.
    Frozen optimizer-facing shards must have no gradients or optimizer state; public
    full-gradient inspection can temporarily materialize flat-buffer views. Every owned
    nonempty trainable shard needs a finite dense FP32 gradient. Do not mutate
    parameters or optimizer membership after construction.
    """

    def __init__(self, *args, frozen_parameter_policy, **kwargs):
        if type(frozen_parameter_policy) is not FrozenFP32Policy:
            raise ValueError("Explicit frozen_fp32_policy is required")
        self.frozen_parameter_policy = frozen_parameter_policy
        self._frozen_policy_sha = frozen_parameter_policy.sha256
        super().__init__(*args, **kwargs)

    def _partition_matches(self):
        current = named(self.model)
        policy = self.frozen_parameter_policy
        return (
            policy.sha256 == self._frozen_policy_sha
            and set(current) == set(policy.trainable_names) | set(policy.frozen_names)
            and all(p.dtype == torch.float32 for p in current.values())
            and all(current[n].requires_grad for n in policy.trainable_names)
            and all(not current[n].requires_grad for n in policy.frozen_names)
        )

    def _validate_master_parameters(self):
        if (
            self.args.bf16
            or self.args.fp16
            or _clip_limit(self.args.max_grad_norm) <= 0
            or not self._partition_matches()
        ):
            raise ValueError(
                "Frozen policy requires its exact FP32 partition and positive clipping"
            )
        self._frozen_initial_shapes = {
            n: list(p.shape) for n, p in named(self.model).items()
        }

    def create_accelerator_and_postprocess(self):
        super().create_accelerator_and_postprocess()
        signature = dict(
            policy=self.frozen_parameter_policy.sha256,
            shapes=self._frozen_initial_shapes,
        )
        signatures = [None] * 2
        dist.all_gather_object(signatures, signature)
        if signatures[0] != signatures[1]:
            raise ValueError("Frozen parameter policy/shape disagreement across ranks")
        original_clip = self.accelerator.clip_grad_norm_

        def checked_clip(parameters, max_norm, norm_type=2):
            self._frozen_clip_ready = False
            result = original_clip(parameters, max_norm, norm_type=norm_type)
            self._frozen_clip_ready = True
            return result

        self.accelerator.clip_grad_norm_ = checked_clip

    def _optimizer_matches(self):
        raw = getattr(self.optimizer, "optimizer", self.optimizer)
        if not isinstance(raw, torch.optim.AdamW):
            return False
        wanted = [p for p in self.model.parameters() if p.requires_grad]
        wanted_ids = {id(p) for p in wanted}
        actual = [p for g in raw.param_groups for p in g["params"]]
        return (
            len(actual) == len(wanted)
            and len({id(p) for p in actual}) == len(actual)
            and {id(p) for p in actual} == wanted_ids
            and all(id(p) in wanted_ids for p in raw.state)
        )

    def _gradient_problem(self, parameters, device):
        bad = torch.tensor(
            int(not self._partition_matches() or not self._optimizer_matches()),
            dtype=torch.int32,
            device=device,
        )
        for p in parameters:
            if not p.requires_grad:
                if p.grad is not None:
                    bad.fill_(1)
                continue
            if p.numel() == 0:
                continue
            g = p.grad
            if (
                g is None
                or g.layout != torch.strided
                or g.dtype != torch.float32
                or g.device != device
            ):
                bad.fill_(1)
            else:
                bad = torch.maximum(bad, (~torch.isfinite(g)).any().to(torch.int32))
        return bad

    def create_scheduler(self, num_training_steps, optimizer=None):
        scheduler = super().create_scheduler(num_training_steps, optimizer=optimizer)
        bad = torch.tensor(
            int(not self._optimizer_matches()), device=self.accelerator.device
        )
        dist.all_reduce(bad, op=dist.ReduceOp.MAX)
        if bad.item():
            raise ValueError(
                "Frozen policy requires AdamW over exactly the trainable parameters"
            )
        raw = getattr(self.optimizer, "optimizer", self.optimizer)
        if getattr(self, "_frozen_gate_optimizer", None) is not raw:

            def gate(optimizer, args, kwargs):
                bad = self._gradient_problem(
                    list(self.model.parameters()), self.accelerator.device
                )
                if not getattr(self, "_frozen_clip_ready", False):
                    bad.fill_(1)
                dist.all_reduce(bad, op=dist.ReduceOp.MAX)
                if bad.item():
                    raise FloatingPointError(
                        "Frozen/trainable gradient or optimizer refusal; reload checkpoint"
                    )
                self._frozen_clip_ready = False
                self._frozen_gate_calls = getattr(self, "_frozen_gate_calls", 0) + 1

            raw.register_step_pre_hook(gate)
            self._frozen_gate_optimizer = raw
        return scheduler

    def _checkpoint_policy(self):
        return dict(
            schema=1,
            precision="FP32",
            policy_sha256=self.frozen_parameter_policy.sha256,
            trainable_names=list(self.frozen_parameter_policy.trainable_names),
            frozen_names=list(self.frozen_parameter_policy.frozen_names),
            shapes=self._frozen_initial_shapes,
        )

    def _save_checkpoint(self, model, trial):
        from pathlib import Path

        bad = torch.tensor(
            int(not self._partition_matches()), device=self.accelerator.device
        )
        dist.all_reduce(bad, op=dist.ReduceOp.MAX)
        if bad.item():
            raise ValueError("Frozen policy changed before checkpoint save")
        super()._save_checkpoint(model, trial)
        root = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
        if self.args.should_save:
            (root / "bc250-frozen-fp32.json").write_text(
                json.dumps(self._checkpoint_policy(), sort_keys=True, indent=2) + "\n"
            )

    def _load_optimizer_and_scheduler(self, checkpoint):
        if checkpoint is not None:
            from pathlib import Path

            path = Path(checkpoint) / "bc250-frozen-fp32.json"
            bad = torch.tensor(int(not path.is_file()), device=self.accelerator.device)
            if path.is_file():
                try:
                    if json.loads(path.read_text()) != self._checkpoint_policy():
                        bad.fill_(1)
                except (ValueError, OSError):
                    bad.fill_(1)
            dist.all_reduce(bad, op=dist.ReduceOp.MAX)
            if bad.item():
                raise ValueError(
                    "Frozen checkpoint policy missing or changed on a rank"
                )
        return super()._load_optimizer_and_scheduler(checkpoint)
