#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Native FSDP global clipping, CPU all-gradient references, refusal and fresh replay."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import uuid

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import BertConfig, BertForMaskedLM, TrainerCallback, TrainingArguments
from bc250_trainer_fsdp import BC250FSDPTrainer, PINS, fsdp_fp16_parameter_policy
from bc250_trainer import masked_token_loss
from check_fsdp_policy_d25 import contracts
from accelerate.utils import GradScalerKwargs
from distributed_d15_trainer import OwnedData, cpu_token_ce, global_relative
from distributed_d05_train import atomic_json
from distributed_d10_checkpoint import digest
from distributed_d10_restart import cpu_tree, exact, fingerprint, rng_state


def main(fixture=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--case",
        choices=[
            "fp16",
            "overflow",
            "inactive",
            "late",
            "scale",
            "fp32-control",
            "bf16-control",
        ]
        + (
            [
                "fp32",
                "bf16",
                "frozen",
                "frozen-missing",
                "frozen-grad",
                "frozen-late",
                "frozen-policy",
            ]
            if fixture is not None
            else []
        ),
        required=True,
    )
    p.add_argument("--action", choices=["seed", "resume"], required=True)
    p.add_argument("--seed-directory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    frozen = bool(getattr(fixture, "frozen", False))
    refusals = {"frozen-missing", "frozen-grad", "frozen-late"}
    r = dict(status="running", checks=[], transcript=[])
    try:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        r["CPU_policy_contracts"] = contracts()
        half = not frozen and a.case not in [
            "fp32-control",
            "bf16-control",
            "fp32",
            "bf16",
        ]
        amp = not frozen and a.case not in ["fp32-control", "fp32"]
        dtype = torch.float16 if half else torch.bfloat16
        bf16 = a.case in ["bf16-control", "bf16"]
        one_step = (
            a.case
            in [
                "inactive",
                "fp32-control",
                "bf16-control",
                "late",
                "scale",
            ]
            or a.case in refusals
        )
        torch.manual_seed(773 if a.action == "seed" else 1773)
        if fixture is None:
            cfg = BertConfig(
                vocab_size=97,
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=64,
                max_position_embeddings=32,
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
                pad_token_id=0,
                use_cache=False,
                attn_implementation="eager",
            )
            base = BertForMaskedLM(cfg)
        else:
            base = fixture.model()
            fixture.check(base)
        base.accepts_loss_kwargs = False
        names = list(dict(base.named_parameters()))
        oracle = copy.deepcopy(base).double()
        oracle_amp = copy.deepcopy(base).float()
        refopt = torch.optim.AdamW(
            (p for p in oracle.parameters() if p.requires_grad),
            lr=0.001,
            eps=1e-5,
            weight_decay=0.01,
            foreach=False,
        )
        data = OwnedData() if fixture is None else fixture.dataset()
        targets = data.labels if fixture is None else data.targets
        token_ce = cpu_token_ce if fixture is None else fixture.cpu_loss
        layer = "BertLayer" if fixture is None else fixture.layer
        if fixture is not None:
            r["CPU_causal_contracts"] = fixture.contracts()
        nparams = sum(p.numel() for p in base.parameters())
        ntrainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
        if fixture is not None:
            r["fixture_source_sha256"] = digest(Path(fixture.__file__))
        args = TrainingArguments(
            output_dir=str(a.output.parent / "trainer-output"),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            max_steps=1 if one_step else 4,
            bf16=bf16,
            fp16=half,
            max_grad_norm=(
                100.0 if a.case == "inactive" else getattr(fixture, "clip_limit", 0.05)
            ),
            learning_rate=0.001,
            report_to="none",
            save_strategy="no" if one_step else "steps",
            save_steps=2,
            save_on_each_node=True,
            save_total_limit=None,
            logging_strategy="no",
            eval_strategy="no",
            disable_tqdm=True,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            train_sampling_strategy="sequential",
            ddp_backend="nccl",
            seed=911,
            data_seed=1011,
            average_tokens_across_devices=True,
            full_determinism=True,
            lr_scheduler_type="constant",
            fsdp="full_shard auto_wrap",
            fsdp_config=dict(
                version=1,
                transformer_layer_cls_to_wrap=[layer],
                use_orig_params=True,
                sync_module_states=True,
                cpu_ram_efficient_loading=False,
                forward_prefetch=False,
                backward_prefetch="NO_PREFETCH",
            ),
            accelerator_config={
                "gradient_accumulation_kwargs": {"adjust_scheduler": False}
            },
        )
        args.fsdp_plugin_args["state_dict_type"] = "SHARDED_STATE_DICT"
        if amp:
            from bc250_trainer_fsdp import fsdp_bf16_parameter_policy

            args.fsdp_plugin_args["mixed_precision_policy"] = (
                fsdp_fp16_parameter_policy() if half else fsdp_bf16_parameter_policy()
            )
        a.seed_directory.mkdir(parents=True, exist_ok=True)
        if frozen:
            f = a.seed_directory / "frozen-initial.pt"
            if a.action == "seed":
                assert not f.exists()
                torch.save(
                    cpu_tree(
                        {
                            n: p
                            for n, p in base.named_parameters()
                            if not p.requires_grad
                        }
                    ),
                    f,
                )
            frozen_initial = torch.load(f, map_location="cpu", weights_only=True)
            r["frozen_initial_digest"] = fingerprint(frozen_initial)
        local = dict(rows=[], losses=[], draws=[], windows=0, registered=False)
        local["counts"], local["local_active"] = [], []
        local["linear_types"] = set()
        local["weight_types"] = set()

        def observe_linear(module, inputs, output):
            local["linear_types"].add(str(output.dtype))
            local["weight_types"].add(str(module.weight.dtype))

        for module in base.modules():
            if isinstance(
                module,
                (torch.nn.Linear,) if fixture is None else fixture.observed_modules,
            ):
                module.register_forward_hook(observe_linear)
        birth = dict(
            uuid=uuid.uuid4().hex,
            pid=os.getpid(),
            start_ticks=Path("/proc/self/stat")
            .read_text()
            .split(") ", 1)[1]
            .split()[19],
        )

        def named(model):
            return {
                name.replace("_fsdp_wrapped_module.", ""): value
                for name, value in model.named_parameters()
            }

        def trainable(model):
            return {n: p for n, p in named(model).items() if p.requires_grad}

        trainer_base = fixture.trainer_class if frozen else BC250FSDPTrainer

        class Observed(trainer_base):
            def _build_accelerator_args(self, **kwargs):
                result = super()._build_accelerator_args(**kwargs)
                if half:
                    result["kwargs_handlers"] = [
                        GradScalerKwargs(
                            init_scale=128 if a.action == "seed" else 8,
                            growth_interval=2,
                        )
                    ]
                return result

            def create_accelerator_and_postprocess(self):
                super().create_accelerator_and_postprocess()
                native_clip = self.accelerator.clip_grad_norm_

                def observed_clip(parameters, max_norm, norm_type=2):
                    parameters = list(parameters)
                    scale = self.accelerator.scaler.get_scale() if half else 1.0
                    local["scale_before"] = scale
                    local["skip"] = a.case == "overflow" and self.state.global_step == 2
                    if (
                        local["skip"]
                        or a.case in ["late", "scale"]
                        or a.case in refusals
                    ):
                        local["before_refusal"] = dict(
                            params=cpu_tree(named(self.model)),
                            optimizer=cpu_tree(self.optimizer.state_dict()),
                            scheduler=cpu_tree(self.lr_scheduler.state_dict()),
                        )
                    if local["skip"]:
                        if self.accelerator.process_index == 1:
                            next(
                                p
                                for p in parameters
                                if p.grad is not None and p.grad.numel()
                            ).grad.view(-1)[0] = float("inf")
                        flags = [None] * 2
                        dist.all_gather_object(
                            flags,
                            any(
                                p.grad is not None
                                and not torch.isfinite(p.grad).all().item()
                                for p in parameters
                            ),
                        )
                        assert flags == [False, True]
                        r["raw_rank_nonfinite_flags"] = flags
                        before = local.get("root_clip_calls", 0)
                        norm = native_clip(parameters, max_norm, norm_type=norm_type)
                        assert (
                            torch.isinf(norm).item()
                            and local.get("root_clip_calls", 0) == before
                        )
                        local["clip"] = dict(
                            norm="inf", CPU_norm=None, coefficient=None
                        )
                        r["checks"].append(
                            "rank1-overflow-native-unscale-both-skip-before-root-clip"
                        )
                        return norm
                    if a.case == "scale" and self.accelerator.process_index == 1:
                        self.accelerator.scaler.update(new_scale=scale / 2)
                    # Public full-gradient snapshot before native root clipping.
                    with FSDP.summon_full_params(
                        self.model, writeback=False, with_grads=True
                    ):
                        if fixture is not None:
                            fixture.check(self.model)
                        local["preclip"] = {
                            n: cpu_tree(p.grad).double() / scale
                            for n, p in trainable(self.model).items()
                        }
                        if frozen:
                            fixture.check_frozen(
                                self.model, frozen_initial, full_gradient_views=True
                            )
                    wanted = math.sqrt(
                        sum(float(g.square().sum()) for g in local["preclip"].values())
                    )
                    coefficient = min(1.0, max_norm / (wanted + 1e-6))
                    assert (
                        (wanted < max_norm and coefficient == 1)
                        if a.case == "inactive"
                        else (wanted > max_norm and coefficient < 1)
                    )
                    before = local.get("root_clip_calls", 0)
                    if (
                        a.case in ["frozen-missing", "frozen-grad"]
                        and self.accelerator.process_index == 1
                    ):
                        p = next(
                            p
                            for p in parameters
                            if p.numel()
                            and p.requires_grad == (a.case == "frozen-missing")
                        )
                        p.grad = (
                            None if a.case == "frozen-missing" else torch.zeros_like(p)
                        )
                    norm = native_clip(parameters, max_norm, norm_type=norm_type)
                    assert local["root_clip_calls"] == before + 1
                    torch.testing.assert_close(
                        norm.cpu().double(),
                        torch.tensor(wanted, dtype=torch.float64),
                        atol=1e-7,
                        rtol=1e-5,
                    )
                    with FSDP.summon_full_params(
                        self.model, writeback=False, with_grads=True
                    ):
                        for n, p in trainable(self.model).items():
                            torch.testing.assert_close(
                                p.grad.cpu().double(),
                                local["preclip"][n] * coefficient,
                                atol=1e-8,
                                rtol=5e-5,
                            )
                    local["clip"] = dict(
                        norm=float(norm), CPU_norm=wanted, coefficient=coefficient
                    )
                    r.setdefault("clip_records", []).append(
                        dict(step=self.state.global_step, **local["clip"])
                    )
                    r["checks"].append(
                        "native-root-global-norm-and-ALL-clipped-gradients-"
                        + str(self.state.global_step)
                    )
                    return norm

                self.accelerator.clip_grad_norm_ = observed_clip

            def create_optimizer(self):
                if self.optimizer is None:
                    self.optimizer = torch.optim.AdamW(
                        (p for p in self.model.parameters() if p.requires_grad),
                        lr=0.001,
                        eps=1e-5,
                        weight_decay=0.01,
                        foreach=False,
                    )
                self.raw_optimizer = self.optimizer
                self.raw_optimizer.register_step_post_hook(
                    lambda *_: local.__setitem__(
                        "raw_updates", local.get("raw_updates", 0) + 1
                    )
                )
                return self.optimizer

            def training_step(self, model, inputs, num_items_in_batch=None):
                assert isinstance(model, FSDP)
                if not local["registered"]:
                    assert sum(isinstance(m, FSDP) for m in model.modules()) == 3
                    if fixture is not None:
                        fixture.check(model)
                    counts = [None] * 2
                    dist.all_gather_object(
                        counts, sum(p.numel() for p in model.parameters())
                    )
                    assert sum(counts) == nparams
                    if frozen:
                        parts = [None] * 2
                        dist.all_gather_object(
                            parts,
                            sum(
                                p.numel() for p in model.parameters() if p.requires_grad
                            ),
                        )
                        assert sum(parts) == ntrainable
                        r["local_trainable_elements"] = parts
                    r["local_shard_parameter_elements"] = counts
                    r["checks"].append(
                        "actual-classic-FSDP-root-and-two-"
                        + layer
                        + "-wraps-original-FP32-shards"
                    )
                    local["registered"] = True
                    root_clip = model.clip_grad_norm_

                    def count_root_clip(*args, **kwargs):
                        local["root_clip_calls"] = local.get("root_clip_calls", 0) + 1
                        return root_clip(*args, **kwargs)

                    model.clip_grad_norm_ = count_root_clip
                rows = (inputs["input_ids"][:, 0] - 1).tolist()
                local["rows"].extend(rows)
                local["draws"].append(
                    [
                        random.random(),
                        float(np.random.random()),
                        torch.rand(()).item(),
                        torch.rand((), device="cuda").item(),
                    ]
                )
                previous = len(local["reductions"])
                local["in_training"] = True
                try:
                    value = super().training_step(model, inputs, num_items_in_batch)
                finally:
                    local["in_training"] = False
                assert len(local["reductions"]) - previous == (
                    (2 if frozen else 3) if self.accelerator.sync_gradients else 0
                )
                assert local["linear_types"] == (
                    {str(dtype)} if amp else {"torch.float32"}
                ) and local["weight_types"] == (
                    {str(dtype)} if amp else {"torch.float32"}
                )
                assert all(
                    p.dtype == torch.float32
                    and (
                        not self.accelerator.sync_gradients
                        or p.grad is None
                        or p.grad.dtype == torch.float32
                    )
                    for p in model.parameters()
                )
                local["losses"].append(float(value))
                return value

            def _load_rng_state(self, checkpoint):
                super()._load_rng_state(checkpoint)
                if checkpoint:
                    actual_state = state()
                    wanted_state = torch.load(
                        a.seed_directory / "checkpoint-state.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                    r["restore_optimizer_actual_keys"] = sorted(
                        actual_state["optimizer"]["state"]
                    )
                    r["restore_optimizer_expected_keys"] = sorted(
                        wanted_state["optimizer"]["state"]
                    )
                    r["restore_parameter_numels"] = [
                        p.numel() for p in self.model.parameters()
                    ]
                    exact(actual_state, wanted_state)
                    if fixture is not None:
                        fixture.check(self.model)
                    r["checks"].append(
                        "fresh-native-FSDP-restore-all-shards-optimizer-scheduler-four-RNG-exact"
                    )

            def _save_checkpoint(self, model, trial):
                super()._save_checkpoint(model, trial)
                if self.state.global_step == 2:
                    torch.save(state(), a.seed_directory / "checkpoint-state.pt")
                    ckpt = Path(self.args.output_dir) / "checkpoint-2"
                    assert (ckpt / "pytorch_model_fsdp_0").is_dir() and (
                        ckpt / "optimizer_0"
                    ).is_dir()
                    if half:
                        assert (ckpt / "scaler.pt").is_file()
                    r["checkpoint_files"] = [
                        dict(
                            name=str(f.relative_to(ckpt)),
                            bytes=f.stat().st_size,
                            sha256=digest(f),
                        )
                        for f in sorted(ckpt.rglob("*"))
                        if f.is_file()
                    ]
                    r["checks"].append(
                        "native-Trainer-sharded-model-optimizer-scheduler-rank-RNG-checkpoint2"
                    )

        def observed_loss(outputs, labels, num_items_in_batch=None):
            assert outputs.logits.dtype == torch.float32
            if fixture is None:
                return masked_token_loss(outputs, labels, num_items_in_batch)
            local["counts"].append(int(num_items_in_batch))
            active = int(labels[:, 1:].ne(-100).sum())
            local["local_active"].append(active)
            value = fixture.loss(outputs, labels, num_items_in_batch)
            if active == 0:
                assert (
                    value.requires_grad
                    and torch.isfinite(value).item()
                    and value.item() == 0
                )
                r["checks"].append(
                    "GPU-local-zero-target-micro-positive-global-zero-loss"
                )
            return value

        trainer = Observed(
            model=base,
            args=args,
            train_dataset=data,
            compute_loss_func=observed_loss,
            **({} if fixture is None else {"data_collator": fixture.collator()}),
            **({} if not frozen else {"frozen_parameter_policy": fixture.policy(base)}),
        )
        acc = trainer.accelerator
        assert acc.native_amp == amp and (acc.scaler is not None) == half
        if half:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            assert type(acc.scaler) is ShardedGradScaler
        native_gather = dist.all_gather_single
        local["gathers"] = []

        def measured_gather(output, input, *positional, **keywords):
            if local.get("in_training"):
                expected = dtype if amp else torch.float32
                assert input.dtype == output.dtype == expected
                local["gathers"].append(
                    dict(
                        input_bytes=input.numel() * input.element_size(),
                        elements=input.numel(),
                        dtype=str(input.dtype),
                    )
                )
            return native_gather(output, input, *positional, **keywords)

        dist.all_gather_single = measured_gather
        native_reduce = dist.reduce_scatter_single
        local["reductions"] = []

        def measured_reduce(output, input, *positional, **keywords):
            assert input.dtype == output.dtype == torch.float32
            local["reductions"].append(
                dict(
                    input_bytes=input.numel() * input.element_size(),
                    dtype=str(input.dtype),
                )
            )
            return native_reduce(output, input, *positional, **keywords)

        dist.reduce_scatter_single = measured_reduce
        assert (
            acc.distributed_type.value == "FSDP"
            and acc.gradient_accumulation_steps == 1
        )
        assert torch.cuda.get_device_properties(0).gcnArchName == "gfx1013:xnack-"
        if frozen:
            r["frozen_helper_sha256"] = digest(
                Path(__file__).with_name("bc250_fsdp_frozen.py")
            )
            r["frozen_policy_sha256"] = trainer.frozen_parameter_policy.sha256
        r.update(
            rank=acc.process_index,
            birth=birth,
            config=dict(
                world=2,
                parameters=nparams,
                trainable_parameters=ntrainable,
                records=28,
                accumulation=2,
                per_device_micro=2,
                precision=(
                    "FP16 working params+allgather/FP32 masters+reduce/native ShardedGradScaler"
                    if half
                    else "BF16 working params" if amp else "FP32"
                ),
                clip_limit=args.max_grad_norm,
                case=a.case,
                source_pins=PINS,
                fixture="Bert MLM" if fixture is None else fixture.name,
                helper_sha256=digest(Path(__file__).with_name("bc250_trainer_fsdp.py")),
            ),
            source_sha256=digest(Path(__file__)),
        )

        def state():
            return dict(
                params=cpu_tree(named(trainer.model)),
                optimizer=cpu_tree(trainer.optimizer.state_dict()),
                scheduler=cpu_tree(trainer.lr_scheduler.state_dict()),
                rng=rng_state(),
                global_step=trainer.state.global_step,
                scaler={} if acc.scaler is None else acc.scaler.state_dict(),
            )

        class Checks(TrainerCallback):
            def on_pre_optimizer_step(self, args, ts, control, **kw):
                batches = [None] * 2
                dist.all_gather_object(batches, local["rows"])
                rows = [i for part in batches for i in part]
                assert sorted(rows) == list(
                    range(ts.global_step * 8, min(ts.global_step * 8 + 8, 28))
                )
                local["grads"] = cpu_tree(
                    {n: p.grad for n, p in named(trainer.model).items()}
                )
                if fixture is not None:
                    count = int(targets[rows, 1:].ne(-100).sum())
                    assert local["counts"] and set(local["counts"]) == {count}
                    assert count == [49, 57, 57, 21][ts.global_step]
                    r.setdefault("target_counts", []).append(
                        dict(
                            step=ts.global_step,
                            shifted=count,
                            raw_unshifted=int(data.labels[rows].ne(-100).sum()),
                            local_active=list(local["local_active"]),
                        )
                    )
                    r["checks"].append(
                        "native-global-shifted-target-count-" + str(ts.global_step)
                    )
                if local["skip"]:
                    r["checks"].append(
                        "native-overflow-window-recorded-for-exact-replay"
                    )
                    return
                with FSDP.summon_full_params(
                    trainer.model, writeback=False, with_grads=True
                ):
                    actual = named(trainer.model)
                    assert set(actual) == set(names)
                    if fixture is not None:
                        fixture.check(trainer.model)
                    if frozen:
                        fixture.check_frozen(
                            trainer.model, frozen_initial, full_gradient_views=True
                        )
                    if a.action == "seed":
                        with torch.no_grad():
                            for name, p in oracle.named_parameters():
                                p.copy_(
                                    actual[name].cpu().to(dtype).double()
                                    if amp
                                    else actual[name].cpu()
                                )
                        refopt.zero_grad(set_to_none=True)
                        target = targets[rows]
                        wanted = token_ce(
                            oracle(
                                input_ids=data.ids[rows], attention_mask=data.mask[rows]
                            ).logits,
                            target,
                            int(target.ne(-100).sum()),
                        )
                        wanted.backward()
                        loss = torch.tensor(sum(local["losses"]), device="cuda")
                        dist.all_reduce(loss)
                        loss /= 2
                        torch.testing.assert_close(
                            loss.cpu().double(),
                            wanted.detach(),
                            atol=(0.001 if half else 0.003) if amp else 3e-6,
                            rtol=(0.001 if half else 0.003) if amp else 5e-5,
                        )
                        if amp:
                            with torch.no_grad():
                                for name, p in oracle_amp.named_parameters():
                                    p.copy_(actual[name].cpu())
                            oracle_amp.to(dtype)
                            denominator = int(target.ne(-100).sum())
                            amp_loss = torch.zeros(())
                            amp_grads = [
                                torch.zeros_like(p, dtype=torch.float32)
                                for p in oracle_amp.parameters()
                            ]
                            for rank_rows in batches:
                                oracle_amp.zero_grad(set_to_none=True)
                                for start in range(0, len(rank_rows), 2):
                                    indices = rank_rows[start : start + 2]
                                    with torch.autocast("cpu", dtype=dtype):
                                        logits = oracle_amp(
                                            input_ids=data.ids[indices],
                                            attention_mask=data.mask[indices],
                                        ).logits
                                    part = token_ce(
                                        logits.float(),
                                        targets[indices],
                                        denominator,
                                    )
                                    (part * 2 * local["scale_before"]).backward()
                                    amp_loss += part.detach()
                                for total, p in zip(amp_grads, oracle_amp.parameters()):
                                    total.add_(
                                        p.grad.float() / (2 * local["scale_before"])
                                    )
                            raw = [
                                local["preclip"][name]
                                for name, _ in oracle.named_parameters()
                            ]
                            high = [p.grad for p in oracle.parameters()]
                            errors = dict(
                                step=ts.global_step,
                                CPU_AMP=global_relative(raw, amp_grads),
                                CPU_FP64=global_relative(raw, high),
                            )
                            assert errors["CPU_AMP"] <= (
                                0.004 if half else 0.02
                            ) and errors["CPU_FP64"] <= (
                                0.006 if half else 0.03
                            ), errors
                            torch.testing.assert_close(
                                loss.cpu(),
                                amp_loss,
                                atol=0.001 if half else 0.003,
                                rtol=0.001 if half else 0.003,
                            )
                            for value, wanted_amp, wanted64 in zip(
                                raw, amp_grads, high
                            ):
                                for expected, bound in [
                                    (wanted_amp, 0.004 if half else 0.02),
                                    (wanted64, 0.006 if half else 0.03),
                                ]:
                                    assert (
                                        value - expected.double()
                                    ).norm().item() <= bound * expected.double().norm().item() + 3e-6 * math.sqrt(
                                        expected.numel()
                                    )
                            r.setdefault("gradient_errors", []).append(errors)
                        else:
                            for name, p in trainable(oracle).items():
                                torch.testing.assert_close(
                                    local["preclip"][name], p.grad, atol=3e-6, rtol=5e-4
                                )
                        r["checks"].append(
                            "CPU-global-loss-and-ALL-raw-gradients-"
                            + str(ts.global_step)
                        )
                        # Restore exact FP32 masters before CPU AdamW, not rounded working weights.
                        for name, p in trainable(oracle).items():
                            with torch.no_grad():
                                p.copy_(actual[name].cpu())
                            p.grad = (
                                local["preclip"][name] * local["clip"]["coefficient"]
                            )
                if a.case in ["late", "frozen-late"] and acc.process_index == 1:
                    next(
                        p
                        for p in trainer.model.parameters()
                        if p.grad is not None and p.grad.numel()
                    ).grad.view(-1)[0] = float("nan")
                assert all(
                    p.numel() == 0 or not p.requires_grad or p.grad is not None
                    for p in trainer.model.parameters()
                )

            def on_optimizer_step(self, args, ts, control, **kw):
                assert acc.optimizer_step_was_skipped == local["skip"]
                if local["skip"]:
                    exact(
                        dict(
                            params=cpu_tree(named(trainer.model)),
                            optimizer=cpu_tree(trainer.optimizer.state_dict()),
                            scheduler=cpu_tree(trainer.lr_scheduler.state_dict()),
                        ),
                        local["before_refusal"],
                    )
                    r["checks"].append(
                        "native-overflow-zero-AdamW-parameter-moment-scheduler-changes"
                    )
                    return
                if a.action == "seed":
                    refopt.step()
                    # Native public full optimizer-state extraction resolves actual original-param shards.
                    fullopt = FSDP.full_optim_state_dict(
                        trainer.model, trainer.raw_optimizer, rank0_only=False
                    )
                    with FSDP.summon_full_params(trainer.model, writeback=False):
                        actual = named(trainer.model)
                        if fixture is not None:
                            fixture.check(trainer.model)
                        if frozen:
                            fixture.check_frozen(trainer.model, frozen_initial)
                            assert set(fullopt["state"]) == set(trainable(oracle))
                        for name, p in trainable(oracle).items():
                            torch.testing.assert_close(
                                actual[name].cpu().double(),
                                p.detach(),
                                atol=4e-6,
                                rtol=5e-5,
                            )
                            for key, atol, rtol in [
                                ("exp_avg", 3e-7, 5e-4),
                                ("exp_avg_sq", 3e-9, 1e-3),
                            ]:
                                torch.testing.assert_close(
                                    fullopt["state"][name][key].cpu().double(),
                                    refopt.state[p][key],
                                    atol=atol,
                                    rtol=rtol,
                                )
                            assert (
                                int(fullopt["state"][name]["step"])
                                == local["raw_updates"]
                            )
                    r["checks"].append(
                        "CPU-AdamW-ALL-parameters-full-moments-and-counters-"
                        + str(ts.global_step)
                    )
                state_elements = sum(
                    v["exp_avg"].numel() for v in trainer.raw_optimizer.state.values()
                )
                local_elements = sum(
                    p.numel() for p in trainer.model.parameters() if p.requires_grad
                )
                assert state_elements == local_elements
                r["optimizer_shard_elements"] = state_elements

            def on_step_end(self, args, ts, control, **kw):
                if frozen:
                    assert trainer._frozen_gate_optimizer is trainer.raw_optimizer
                    assert trainer._frozen_gate_calls == local["raw_updates"]
                    r["frozen_gate_calls"] = trainer._frozen_gate_calls
                if half:
                    assert trainer._fp16_gate_optimizer is trainer.raw_optimizer
                    assert getattr(trainer, "_fp16_gate_calls", 0) == local.get(
                        "raw_updates", 0
                    )
                successful = ts.global_step - (
                    1 if a.case == "overflow" and ts.global_step >= 3 else 0
                )
                assert trainer.lr_scheduler.state_dict()[
                    "last_epoch"
                ] == successful and math.isclose(
                    trainer.optimizer.param_groups[0]["lr"], 0.001
                )
                if half:
                    expected_scale = 128 * 2 ** (ts.global_step // 2)
                    expected_tracker = ts.global_step % 2
                    if a.case == "overflow" and ts.global_step >= 3:
                        expected_scale, expected_tracker = 128, ts.global_step - 3
                    assert acc.scaler.get_scale() == expected_scale
                    assert (
                        acc.scaler.state_dict()["_growth_tracker"] == expected_tracker
                    )
                value = dict(
                    state=state(),
                    grads=local["grads"],
                    rows=local["rows"],
                    losses=local["losses"],
                    draws=local["draws"],
                    clip=local["clip"],
                )
                if fixture is not None:
                    fixture.check(trainer.model)
                    value.update(
                        target_counts=list(local["counts"]),
                        local_active=list(local["local_active"]),
                    )
                if ts.global_step > 2:
                    f = a.seed_directory / f"reference-{ts.global_step}.pt"
                    if a.action == "seed":
                        torch.save(value, f)
                    else:
                        exact(
                            value, torch.load(f, map_location="cpu", weights_only=True)
                        )
                        r["checks"].append(
                            "fresh-FSDP-replay-exact-all-state-window-"
                            + str(ts.global_step)
                        )
                r["transcript"].append(
                    dict(
                        step=ts.global_step,
                        digest=fingerprint(value),
                        rows=list(local["rows"]),
                    )
                )
                local["counts"], local["local_active"] = [], []
                local["rows"] = []
                local["losses"] = []
                local["draws"] = []
                local["windows"] += 1

        trainer.add_callback(Checks())
        try:
            trainer.train(
                resume_from_checkpoint=(
                    str(a.seed_directory / "trainer-output/checkpoint-2")
                    if a.action == "resume"
                    else None
                )
            )
        except ValueError as error:
            if a.case != "frozen-policy" or "Frozen checkpoint policy" not in str(
                error
            ):
                raise
            assert local.get("raw_updates", 0) == 0 and not trainer.raw_optimizer.state
            assert trainer.lr_scheduler.state_dict()["last_epoch"] == 0
            r["checks"].extend(
                [
                    "both-live-ranks-refuse-changed-checkpoint-policy",
                    "zero-AdamW-updates-and-unrestored-optimizer-scheduler",
                ]
            )
            r.update(status="passed", raw_updates=0, refusal=str(error))
            dist.barrier()
            return
        except FloatingPointError:
            if a.case not in ["late", "scale"] and a.case not in refusals:
                raise
            exact(
                dict(
                    params=cpu_tree(named(trainer.model)),
                    optimizer=cpu_tree(trainer.optimizer.state_dict()),
                    scheduler=cpu_tree(trainer.lr_scheduler.state_dict()),
                ),
                local["before_refusal"],
            )
            assert (
                local.get("raw_updates", 0) == 0
                and local.get("root_clip_calls", 0)
                == (1 if a.case in ["late", "frozen-late"] else 0)
                and trainer.state.global_step == 0
            )
            r["checks"].extend(
                [
                    "both-live-ranks-refuse-before-AdamW",
                    "parameters-optimizer-scheduler-unchanged-after-refusal",
                ]
            )
            r.update(status="passed", raw_updates=0)
            dist.barrier()
            return
        assert a.case in [
            "fp16",
            "overflow",
            "inactive",
            "fp32-control",
            "bf16-control",
            "fp32",
            "bf16",
            "frozen",
        ]
        assert trainer.state.global_step == (1 if one_step else 4) and local[
            "windows"
        ] == ((1 if one_step else 4) if a.action == "seed" else 2)
        assert local["root_clip_calls"] == local["windows"] - (
            1 if a.case == "overflow" else 0
        )
        if a.action == "resume":
            old = json.loads((a.seed_directory / "rank.json").read_text())
            assert old["birth"] != birth
            assert [v["digest"] for v in r["transcript"]] == [
                v["digest"] for v in old["transcript"] if v["step"] > 2
            ]
            r["checks"].append(
                "fresh-process-different-birth-and-exact-resumed-transcript"
            )
        assert local["gathers"]
        r["parameter_allgathers"] = local["gathers"]
        r["observed_dtypes"] = dict(
            linear_outputs=sorted(local["linear_types"]),
            forward_weights=sorted(local["weight_types"]),
            reduction_dtype="torch.float32",
            reduction_calls=len(local["reductions"]),
            reduction_input_bytes=sum(v["input_bytes"] for v in local["reductions"]),
            scaler=None if acc.scaler is None else type(acc.scaler).__name__,
            final_scaler={} if acc.scaler is None else acc.scaler.state_dict(),
        )
        r.update(status="passed", final_state_digest=fingerprint(state()))
        if fixture is not None:
            r["tied_parameter_checks"] = fixture.check_calls
        dist.barrier()
    except BaseException as e:
        r.update(status="failed", error=repr(e))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        atomic_json(a.output, r)


if __name__ == "__main__":
    main()
