#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Actual Trainer.evaluate precision scope, encoder AMP and GPU exception restoration."""
import argparse
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import BertConfig, BertForMaskedLM, TrainingArguments
from bc250_trainer_clipping import BC250ClippedTrainer
from bc250_trainer import masked_token_loss
from bc250_compact_token_metrics import CompactTokenMetrics
from bc250_eval_precision import fp32_eval_head
from distributed_d19_evaluate import OwnedData, pad_records
from distributed_d05_train import atomic_json
from distributed_d10_checkpoint import digest
from distributed_d10_restart import cpu_tree, exact


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=["bertwide-bf16"], required=True)
    p.add_argument("--action", choices=["scope"], required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = dict(status="running", checks=[])
    try:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(773)
        cfg = BertConfig(
            vocab_size=32768,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=128,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            pad_token_id=0,
            use_cache=False,
            attn_implementation="eager",
        )
        model = BertForMaskedLM(cfg)
        model.accepts_loss_kwargs = False
        args = TrainingArguments(
            output_dir=str(a.output.parent / "trainer-output"),
            per_device_eval_batch_size=2,
            bf16=True,
            report_to="none",
            disable_tqdm=True,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            ddp_backend="nccl",
            seed=911,
            data_seed=1011,
            average_tokens_across_devices=True,
            full_determinism=True,
            eval_accumulation_steps=2,
            batch_eval_metrics=False,
            eval_do_concat_batches=True,
        )
        metric = CompactTokenMetrics(chunk_tokens=256)
        trainer = BC250ClippedTrainer(
            model=model,
            args=args,
            eval_dataset=OwnedData(wide=True),
            data_collator=pad_records,
            compute_loss_func=masked_token_loss,
            compute_metrics=metric,
            preprocess_logits_for_metrics=metric.preprocess,
        )
        assert (
            trainer.accelerator.num_processes == 2
            and trainer.accelerator.state.backend == "nccl"
        )
        assert all(p.device.type == "cuda" for p in model.parameters())
        before = cpu_tree(model.state_dict())
        ids = [id(p) for p in model.parameters()]
        keys = list(model.state_dict())
        assert (
            model.cls.predictions.decoder.weight
            is model.bert.embeddings.word_embeddings.weight
        )
        query_types = []
        head_types = []
        model.bert.encoder.layer[0].attention.self.query.register_forward_hook(
            lambda m, x, y: query_types.append(str(y.dtype))
        )
        model.cls.register_forward_hook(lambda m, x, y: head_types.append(str(y.dtype)))
        native = trainer.evaluate()
        assert abs(native["eval_token_nll"] - 10.432745588959701) < 1e-11
        assert set(query_types) == set(head_types) == {"torch.bfloat16"}
        r["checks"].append("native-Trainer-evaluate-matches-frozen-D20-predict-metrics")
        query_types.clear()
        head_types.clear()
        with fp32_eval_head(model.cls):
            qualified = trainer.evaluate()
        assert (
            abs(qualified["eval_token_nll"] - 10.43274511247336) < 1e-11
            and qualified["eval_active_tokens"] == 838
        )
        assert set(query_types) == {"torch.bfloat16"} and set(head_types) == {
            "torch.float32"
        }
        r["checks"].append(
            "actual-Trainer-evaluate-FP32-head-metrics-and-observed-BF16-encoder"
        )
        # Exercise scope restoration on a GPU forward followed by an owned exception.
        try:
            with torch.no_grad(), torch.autocast(
                "cuda", dtype=torch.bfloat16
            ), fp32_eval_head(model.cls):
                assert (
                    model.cls(torch.zeros(1, 1, 32, device="cuda")).dtype
                    == torch.float32
                )
                raise RuntimeError("owned-D20-GPU-exception")
        except RuntimeError as e:
            assert str(e) == "owned-D20-GPU-exception"
        assert "forward" not in model.cls.__dict__ and not hasattr(
            model.cls, "_bc250_fp32_eval_scope"
        )
        query_types.clear()
        head_types.clear()
        restored = trainer.evaluate()
        assert set(query_types) == set(head_types) == {"torch.bfloat16"}
        for key in native:
            if key in [
                "eval_loss",
                "eval_token_nll",
                "eval_token_perplexity",
                "eval_token_accuracy",
                "eval_active_tokens",
                "eval_evaluated_examples",
            ]:
                assert native[key] == restored[key]
        r["checks"].append(
            "GPU-exception-restores-original-head-and-exact-native-evaluation-metrics"
        )
        exact(cpu_tree(model.state_dict()), before)
        assert keys == list(model.state_dict()) and ids == [
            id(p) for p in model.parameters()
        ]
        assert (
            model.cls.predictions.decoder.weight
            is model.bert.embeddings.word_embeddings.weight
            and all(p.grad is None for p in model.parameters())
        )
        r["checks"].append(
            "model-state-keys-parameter-identities-tied-weights-and-gradients-preserved"
        )
        values = dict(
            native_token_nll=native["eval_token_nll"],
            head_token_nll=qualified["eval_token_nll"],
        )
        peers = [None] * 2
        dist.all_gather_object(peers, values)
        assert peers[0] == peers[1]
        r["checks"].append("both-ranks-identical-scope-evaluation-metrics")
        r.update(
            status="passed",
            rank=trainer.accelerator.process_index,
            source_sha256=digest(Path(__file__)),
            helpers={
                n: digest(Path(__file__).with_name(n))
                for n in [
                    "bc250_eval_precision.py",
                    "bc250_compact_token_metrics.py",
                    "distributed_d19_evaluate.py",
                ]
            },
            metrics=values,
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
