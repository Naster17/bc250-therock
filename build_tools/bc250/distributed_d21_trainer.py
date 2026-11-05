#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Actual Trainer classic FSDP FP32 token training, CPU references and native restart."""
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
from bc250_trainer_fsdp import BC250FSDPTrainer, PINS
from bc250_trainer import masked_token_loss
from distributed_d15_trainer import OwnedData, cpu_token_ce
from distributed_d05_train import atomic_json
from distributed_d10_checkpoint import digest
from distributed_d10_restart import cpu_tree, exact, fingerprint, rng_state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=["fp32"], required=True)
    p.add_argument("--action", choices=["seed", "resume"], required=True)
    p.add_argument("--seed-directory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = dict(status="running", checks=[], transcript=[])
    try:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(773 if a.action == "seed" else 1773)
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
        base.accepts_loss_kwargs = False
        names = list(dict(base.named_parameters()))
        oracle = copy.deepcopy(base).double()
        refopt = torch.optim.AdamW(
            oracle.parameters(), lr=0.001, eps=1e-5, weight_decay=0.01, foreach=False
        )
        data = OwnedData()
        nparams = sum(p.numel() for p in base.parameters())
        args = TrainingArguments(
            output_dir=str(a.output.parent / "trainer-output"),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            max_steps=4,
            max_grad_norm=0,
            learning_rate=0.001,
            report_to="none",
            save_strategy="steps",
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
                transformer_layer_cls_to_wrap=["BertLayer"],
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
        a.seed_directory.mkdir(parents=True, exist_ok=True)
        local = dict(rows=[], losses=[], draws=[], windows=0, registered=False)
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

        class Observed(BC250FSDPTrainer):
            def create_optimizer(self):
                if self.optimizer is None:
                    self.optimizer = torch.optim.AdamW(
                        self.model.parameters(),
                        lr=0.001,
                        eps=1e-5,
                        weight_decay=0.01,
                        foreach=False,
                    )
                self.raw_optimizer = self.optimizer
                return self.optimizer

            def training_step(self, model, inputs, num_items_in_batch=None):
                assert isinstance(model, FSDP)
                if not local["registered"]:
                    assert sum(isinstance(m, FSDP) for m in model.modules()) == 3
                    counts = [None] * 2
                    dist.all_gather_object(
                        counts, sum(p.numel() for p in model.parameters())
                    )
                    r["local_shard_parameter_elements"] = counts
                    r["checks"].append(
                        "actual-classic-FSDP-root-and-two-BertLayer-wraps-original-FP32-shards"
                    )
                    local["registered"] = True
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
                value = super().training_step(model, inputs, num_items_in_batch)
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

        trainer = Observed(
            model=base,
            args=args,
            train_dataset=data,
            compute_loss_func=masked_token_loss,
        )
        acc = trainer.accelerator
        assert (
            acc.distributed_type.value == "FSDP"
            and acc.gradient_accumulation_steps == 1
        )
        assert torch.cuda.get_device_properties(0).gcnArchName == "gfx1013:xnack-"
        r.update(
            rank=acc.process_index,
            birth=birth,
            config=dict(
                world=2,
                parameters=nparams,
                records=28,
                accumulation=2,
                per_device_micro=2,
                precision="FP32",
                source_pins=PINS,
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
                with FSDP.summon_full_params(
                    trainer.model, writeback=False, with_grads=True
                ):
                    actual = named(trainer.model)
                    assert set(actual) == set(names)
                    if a.action == "seed":
                        with torch.no_grad():
                            for name, p in oracle.named_parameters():
                                p.copy_(actual[name].cpu())
                        refopt.zero_grad(set_to_none=True)
                        target = data.labels[rows]
                        wanted = cpu_token_ce(
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
                            loss.cpu().double(), wanted.detach(), atol=3e-6, rtol=5e-5
                        )
                        for name, p in oracle.named_parameters():
                            torch.testing.assert_close(
                                actual[name].grad.cpu().double(),
                                p.grad,
                                atol=3e-6,
                                rtol=5e-4,
                            )
                        r["checks"].append(
                            "CPU-FP64-global-token-loss-and-ALL-full-gradients-"
                            + str(ts.global_step)
                        )
                        # AdamW oracle consumes admitted actual gradients to isolate optimizer arithmetic.
                        for name, p in oracle.named_parameters():
                            p.grad = actual[name].grad.detach().cpu().double().clone()
                assert all(
                    p.numel() == 0 or p.grad is not None
                    for p in trainer.model.parameters()
                )

            def on_optimizer_step(self, args, ts, control, **kw):
                if a.action == "seed":
                    refopt.step()
                    # Native public full optimizer-state extraction resolves actual original-param shards.
                    fullopt = FSDP.full_optim_state_dict(
                        trainer.model, trainer.raw_optimizer, rank0_only=False
                    )
                    with FSDP.summon_full_params(trainer.model, writeback=False):
                        actual = named(trainer.model)
                        for name, p in oracle.named_parameters():
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
                                == ts.global_step + 1
                            )
                    r["checks"].append(
                        "CPU-AdamW-ALL-parameters-full-moments-and-counters-"
                        + str(ts.global_step)
                    )
                state_elements = sum(
                    v["exp_avg"].numel() for v in trainer.raw_optimizer.state.values()
                )
                local_elements = sum(p.numel() for p in trainer.model.parameters())
                assert state_elements == local_elements
                r["optimizer_shard_elements"] = state_elements

            def on_step_end(self, args, ts, control, **kw):
                assert trainer.lr_scheduler.state_dict()[
                    "last_epoch"
                ] == ts.global_step and math.isclose(
                    trainer.optimizer.param_groups[0]["lr"], 0.001
                )
                value = dict(
                    state=state(),
                    grads=local["grads"],
                    rows=local["rows"],
                    losses=local["losses"],
                    draws=local["draws"],
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
                local["rows"] = []
                local["losses"] = []
                local["draws"] = []
                local["windows"] += 1

        trainer.add_callback(Checks())
        trainer.train(
            resume_from_checkpoint=(
                str(a.seed_directory / "trainer-output/checkpoint-2")
                if a.action == "resume"
                else None
            )
        )
        assert trainer.state.global_step == 4 and local["windows"] == (
            4 if a.action == "seed" else 2
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
        r.update(status="passed", final_state_digest=fingerprint(state()))
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
