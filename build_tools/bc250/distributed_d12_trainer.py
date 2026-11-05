#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Owned actual Trainer DDP CPU references and native fixed-world checkpoint replay."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import uuid

import numpy as np
import torch
import torch.distributed as dist
from transformers import (
    BertConfig,
    BertForMaskedLM,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
import transformers
import accelerate
import bc250_distributed_identity
from bc250_trainer import BC250Trainer, masked_token_loss
from distributed_d05_train import atomic_json
from distributed_d10_checkpoint import digest
from distributed_d10_restart import cpu_tree, exact, fingerprint, rng_state


class OwnedData(torch.utils.data.Dataset):
    def __init__(self, token_weighted):
        self.ids = (torch.arange(28 * 12).reshape(28, 12) * 7 + 3) % 96 + 1
        self.ids[:, 0] = torch.arange(28) + 1
        self.labels = (self.ids * 3 + 7) % 97
        if token_weighted:
            for i in range(28):
                self.labels[i, (i % 11) + 1 :] = -100

    def __len__(self):
        return 28

    def __getitem__(self, i):
        return dict(input_ids=self.ids[i], labels=self.labels[i])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["stock", "mean", "tokens"], required=True)
    parser.add_argument("--action", choices=["seed", "resume"], required=True)
    parser.add_argument("--seed-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    r = dict(status="running", case=a.case, action=a.action, checks=[], transcript=[])
    try:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        cfg = BertConfig(
            vocab_size=97,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=32,
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0,
            pad_token_id=0,
            attn_implementation="eager",
        )
        torch.manual_seed(773 if a.action == "seed" else 1773)
        base = BertForMaskedLM(cfg)
        # Bert's MLM loss ignores num_items_in_batch despite accepting **kwargs.
        # Explicit mean policy uses equal active counts; tokens uses the CE hook.
        base.accepts_loss_kwargs = False
        oracle = copy.deepcopy(base).double()
        optimizer = torch.optim.AdamW(
            base.parameters(), lr=0.001, eps=1e-5, weight_decay=0.01, foreach=False
        )
        refopt = torch.optim.AdamW(
            oracle.parameters(), lr=0.001, eps=1e-5, weight_decay=0.01, foreach=False
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
        data = OwnedData(a.case == "tokens")
        args = TrainingArguments(
            output_dir=str(a.output.parent / "trainer-output"),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            max_steps=1 if a.case == "stock" else 4,
            learning_rate=0.001,
            max_grad_norm=0,
            weight_decay=0.01,
            report_to="none",
            save_strategy="no" if a.case == "stock" else "steps",
            save_steps=2,
            save_on_each_node=True,
            save_total_limit=None,
            logging_strategy="no",
            eval_strategy="no",
            disable_tqdm=True,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            train_sampling_strategy="sequential",
            ddp_find_unused_parameters=False,
            ddp_backend="nccl",
            seed=911,
            data_seed=1011,
            average_tokens_across_devices=True,
            full_determinism=True,
            accelerator_config={
                "split_batches": False,
                "gradient_accumulation_kwargs": {"adjust_scheduler": False},
            },
        )
        birth = dict(
            uuid=uuid.uuid4().hex,
            pid=os.getpid(),
            start_ticks=Path("/proc/self/stat")
            .read_text()
            .split(") ", 1)[1]
            .split()[19],
        )
        raw_updates = [0]
        optimizer.register_step_post_hook(
            lambda *_: raw_updates.__setitem__(0, raw_updates[0] + 1)
        )
        local = dict(
            rows=[],
            losses=[],
            draws=[],
            traffic=dict(calls=0, bytes=0),
            registered=False,
            windows=0,
        )
        stock = a.case == "stock"
        parent = Trainer if stock else BC250Trainer

        class ObservedTrainer(parent):
            def training_step(self, model, inputs, num_items_in_batch=None):
                if not local["registered"]:
                    assert isinstance(model, torch.nn.parallel.DistributedDataParallel)
                    from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import (
                        allreduce_hook,
                    )

                    def hook(state, bucket):
                        state["calls"] += 1
                        state["bytes"] += (
                            bucket.buffer().numel() * bucket.buffer().element_size()
                        )
                        assert bucket.buffer().dtype == torch.float32
                        return allreduce_hook(None, bucket)

                    model.register_comm_hook(local["traffic"], hook)
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
                before = dict(local["traffic"])
                value = super().training_step(model, inputs, num_items_in_batch)
                local["losses"].append(float(value))
                final = self.accelerator.sync_gradients
                assert local["traffic"]["bytes"] - before["bytes"] == (
                    22561 * 4 if final else 0
                )
                if not final:
                    r["checks"].append("Trainer-nonfinal-micro-no-gradient-reduction")
                return value

            def _load_rng_state(self, checkpoint):
                super()._load_rng_state(checkpoint)
                if checkpoint:
                    expected = torch.load(
                        a.seed_directory / "checkpoint-state.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                    exact(state(), expected)
                    r["checks"].append(
                        "native-Trainer-loaded-model-AdamW-StepLR-four-RNG-step-exact"
                    )

            def _save_checkpoint(self, model, trial):
                super()._save_checkpoint(model, trial)
                if self.state.global_step == 2:
                    torch.save(state(), a.seed_directory / "checkpoint-state.pt")
                    checkpoint = Path(self.args.output_dir) / "checkpoint-2"
                    files = [
                        dict(name=p.name, bytes=p.stat().st_size, sha256=digest(p))
                        for p in sorted(checkpoint.iterdir())
                        if p.is_file()
                    ]
                    names = {v["name"] for v in files}
                    assert {
                        "model.safetensors",
                        "optimizer.pt",
                        "scheduler.pt",
                        "trainer_state.json",
                        f"rng_state_{self.args.process_index}.pth",
                    }.issubset(names)
                    atomic_json(
                        a.seed_directory / "complete.json",
                        dict(
                            config=config,
                            rank=self.args.process_index,
                            birth=birth,
                            files=files,
                        ),
                    )
                    r["checks"].append(
                        "native-Trainer-checkpoint2-all-files-on-each-host"
                    )

        trainer = ObservedTrainer(
            model=base,
            args=args,
            train_dataset=data,
            optimizers=(optimizer, scheduler),
            compute_loss_func=masked_token_loss if a.case == "tokens" else None,
        )
        acc = trainer.accelerator
        rank = acc.process_index
        assert (
            acc.num_processes == 2
            and acc.distributed_type.value == "MULTI_GPU"
            and acc.state.backend == "nccl"
        )
        assert acc.gradient_accumulation_steps == (2 if stock else 1)
        assert torch.cuda.get_device_properties(0).gcnArchName == "gfx1013:xnack-"
        tool = Path(bc250_distributed_identity.__file__).parent
        package = Path(transformers.__file__).parent
        config = dict(
            case=a.case,
            world=2,
            parameters=sum(v.numel() for v in base.parameters()),
            global_batch=8,
            accumulation=2,
            records=28,
            workers=0,
            transformers=transformers.__version__,
            accelerate=accelerate.__version__,
            fixture=digest(Path(__file__)),
            compat=digest(Path(__file__).with_name("bc250_trainer.py")),
            native=digest(tool / "torch-identity.json"),
            trainer=digest(package / "trainer.py"),
            bert=digest(package / "models/bert/modeling_bert.py"),
        )
        r.update(rank=rank, birth=birth, config=config, source_sha256=config["fixture"])

        def state():
            return dict(
                params=[cpu_tree(v) for v in trainer.model.parameters()],
                optimizer=cpu_tree(trainer.optimizer.state_dict()),
                scheduler=cpu_tree(trainer.lr_scheduler.state_dict()),
                rng=rng_state(),
                global_step=trainer.state.global_step,
            )

        class Checks(TrainerCallback):
            def on_pre_optimizer_step(self, args, ts, control, **kwargs):
                step = ts.global_step
                batches = [None] * 2
                dist.all_gather_object(batches, local["rows"])
                rows = [i for part in batches for i in part]
                expected_rows = list(range(step * 8, min(step * 8 + 8, 28)))
                assert sorted(rows) == expected_rows, (rows, expected_rows)
                if a.action == "seed":
                    refopt.zero_grad(set_to_none=True)
                    output = oracle(input_ids=data.ids[rows])
                    wanted = torch.nn.functional.cross_entropy(
                        output.logits.reshape(-1, 97),
                        data.labels[rows].reshape(-1),
                        ignore_index=-100,
                    )
                    wanted.backward()
                    total = torch.tensor(sum(local["losses"]), device="cuda")
                    dist.all_reduce(total)
                    total /= 2
                    # Returned loss is normalized in Trainer but NOT by Accelerator.
                    torch.testing.assert_close(
                        total.cpu().double(), wanted.detach(), atol=3e-6, rtol=5e-5
                    )
                    for actual, desired in zip(
                        trainer.model.parameters(), oracle.parameters()
                    ):
                        torch.testing.assert_close(
                            actual.grad.cpu().double(),
                            desired.grad * (0.5 if stock else 1),
                            atol=3e-6,
                            rtol=5e-4,
                        )
                    r["checks"].append(
                        "stock-Trainer-double-division-reproduced"
                        if stock
                        else f"CPU-global-token-loss-all-gradients-{step}"
                    )
                local["grads"] = [cpu_tree(v.grad) for v in trainer.model.parameters()]

            def on_optimizer_step(self, args, ts, control, **kwargs):
                if a.action == "seed" and not stock:
                    refopt.step()
                    for actual, desired in zip(
                        trainer.model.parameters(), oracle.parameters()
                    ):
                        torch.testing.assert_close(
                            actual.detach().cpu().double(),
                            desired,
                            atol=4e-6,
                            rtol=5e-5,
                        )
                        for k, atol, rtol in [
                            ("exp_avg", 3e-7, 5e-4),
                            ("exp_avg_sq", 3e-9, 1e-3),
                        ]:
                            torch.testing.assert_close(
                                optimizer.state[actual][k].cpu().double(),
                                refopt.state[desired][k],
                                atol=atol,
                                rtol=rtol,
                            )
                    # Independent scheduler driven by one completed update.
                    for group in refopt.param_groups:
                        group["lr"] = 0.001 * 0.8 ** ((ts.global_step + 1) // 2)
                    r["checks"].append(f"CPU-AdamW-all-params-moments-{ts.global_step}")

            def on_step_end(self, args, ts, control, **kwargs):
                assert trainer.lr_scheduler.state_dict()["last_epoch"] == ts.global_step
                assert trainer.optimizer.param_groups[0]["lr"] == 0.001 * 0.8 ** (
                    ts.global_step // 2
                )
                if not stock:
                    value = dict(
                        state=state(),
                        grads=local["grads"],
                        rows=local["rows"],
                        losses=local["losses"],
                        draws=local["draws"],
                    )
                    if ts.global_step > 2:
                        path = a.seed_directory / f"reference-{ts.global_step}.pt"
                        if a.action == "seed":
                            torch.save(value, path)
                        else:
                            exact(
                                value,
                                torch.load(path, map_location="cpu", weights_only=True),
                            )
                            r["checks"].append(
                                f"fresh-Trainer-replay-exact-all-state-{ts.global_step}"
                            )
                    r["transcript"].append(
                        dict(
                            step=ts.global_step,
                            digest=fingerprint(value),
                            rows=list(local["rows"]),
                        )
                    )
                local["windows"] += 1
                local["rows"] = []
                local["draws"] = []
                local["losses"] = []

        trainer.add_callback(Checks())
        checkpoint = a.seed_directory / "trainer-output/checkpoint-2"
        if a.action == "resume":
            header = json.loads((a.seed_directory / "complete.json").read_text())
            assert header["rank"] == rank and header["config"] == config
            for f in header["files"]:
                path = checkpoint / f["name"]
                assert path.stat().st_size == f["bytes"] and digest(path) == f["sha256"]
            assert (
                header["birth"]["uuid"] != birth["uuid"]
                and header["birth"]["start_ticks"] != birth["start_ticks"]
            )
            r["source_birth"] = header["birth"]
        outcome = trainer.train(
            resume_from_checkpoint=str(checkpoint) if a.action == "resume" else None
        )
        assert outcome.global_step == (1 if stock else 4)
        assert raw_updates[0] == (1 if stock else 4 if a.action == "seed" else 2)
        r["checks"].append(
            "actual-Trainer-DDP-loader-accumulation-AdamW-StepLR-completed"
        )
        r.update(
            final_digest=fingerprint(state()),
            optimizer_updates=raw_updates[0],
            global_step=trainer.state.global_step,
            status="passed",
        )
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
