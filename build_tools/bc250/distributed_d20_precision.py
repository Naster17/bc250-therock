#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Diagnose the frozen wide Bert BF16 logit gate without changing its tolerance."""
import argparse
import copy
import gc
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from transformers import BertConfig, BertForMaskedLM, TrainingArguments
from bc250_trainer_clipping import BC250ClippedTrainer
from bc250_trainer import masked_token_loss
from bc250_compact_token_metrics import CompactTokenMetrics
from bc250_array_digest import array_digest
from check_array_digest_d20 import contracts
from check_eval_precision_d20 import contracts as head_contracts
from bc250_eval_precision import fp32_eval_head
from distributed_d19_evaluate import OwnedData, pad_records
from distributed_d05_train import atomic_json
from distributed_d10_checkpoint import digest
from distributed_d10_restart import cpu_tree, exact


def comparison(actual, reference, mask, *, atol=0.003, rtol=0.003):
    """Chunk CPU comparisons to avoid whole-dataset FP64 copies/large Python lists."""
    count = 0
    total = 0
    maximum = 0.0
    err2 = 0.0
    ref2 = 0.0
    worst = []
    for row in range(len(actual)):
        n = int(mask[row].sum())
        x = torch.as_tensor(actual[row, :n]).double()
        y = torch.as_tensor(reference[row, :n]).double()
        delta = (x - y).abs()
        invalid = delta.gt(atol + rtol * y.abs())
        count += int(invalid.sum())
        total += x.numel()
        maximum = max(maximum, float(delta.max()))
        err2 += float(delta.square().sum())
        ref2 += float(y.square().sum())
        # Store only outliers, bounded even if a whole row is bad.
        for pos, col in invalid.nonzero()[:8].tolist():
            worst.append(
                dict(
                    row=row,
                    position=pos,
                    vocab=col,
                    actual=float(x[pos, col]),
                    reference=float(y[pos, col]),
                    absolute_error=float(delta[pos, col]),
                )
            )
    worst = sorted(worst, key=lambda v: v["absolute_error"], reverse=True)[:8]
    return dict(
        compared_values=total,
        violations=count,
        max_absolute_error=maximum,
        relative_L2=(err2 / ref2) ** 0.5,
        atol=atol,
        rtol=rtol,
        outliers=worst,
    )


def matched_cpu(model, data, *, autocast):
    """Recreate both native distributed rank microbatches, including duplicate tail."""
    output = torch.full((19, 64, 32768), -100.0, dtype=torch.float32)
    for step in range(5):
        for rank in range(2):
            rows = [(step * 4 + rank * 2 + i) % 19 for i in range(2)]
            batch = pad_records([data[i] for i in rows])
            batch.pop("labels")
            with torch.no_grad(), torch.autocast(
                "cpu", dtype=torch.bfloat16, enabled=autocast
            ):
                z = model(**batch).logits.float()
            for i, row in enumerate(rows):
                n = int(data.mask[row].sum())
                if step == 4 and rank == 1 and i == 1:
                    continue  # native gather_for_metrics de-duplicates row0
                output[row, :n] = z[i, :n]
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["bertwide-bf16"], required=True)
    parser.add_argument("--action", choices=["diagnose", "fp32-head"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    r = dict(status="running", checks=[])
    scopes = ExitStack()
    try:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        r["digest_contracts"] = contracts()
        r["head_contracts"] = head_contracts()
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
        base = BertForMaskedLM(cfg)
        base.accepts_loss_kwargs = False
        cpu = copy.deepcopy(base).float().eval()
        oracle = copy.deepcopy(base).double().eval()
        before = cpu_tree(base.state_dict())
        data = OwnedData(wide=True)
        metrics = CompactTokenMetrics(causal=False, chunk_tokens=256)
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
        trainer = BC250ClippedTrainer(
            model=base,
            args=args,
            eval_dataset=data,
            data_collator=pad_records,
            compute_loss_func=masked_token_loss,
            compute_metrics=metrics,
            preprocess_logits_for_metrics=metrics.preprocess,
        )
        if a.action == "fp32-head":
            base.eval()
            scopes.enter_context(fp32_eval_head(base.cls))
            scopes.enter_context(fp32_eval_head(cpu.cls))
        r["policy"] = a.action
        acc = trainer.accelerator
        assert acc.num_processes == 2 and acc.state.backend == "nccl"
        assert all(p.device.type == "cuda" for p in base.parameters())
        assert torch.cuda.get_device_properties(0).gcnArchName == "gfx1013:xnack-"
        r.update(
            rank=acc.process_index,
            source_sha256=digest(Path(__file__)),
            helpers={
                name: digest(Path(__file__).with_name(name))
                for name in [
                    "bc250_array_digest.py",
                    "check_array_digest_d20.py",
                    "bc250_compact_token_metrics.py",
                    "distributed_d19_evaluate.py",
                    "bc250_eval_precision.py",
                    "check_eval_precision_d20.py",
                ]
            },
        )
        # Gather full BF16 logits through native Trainer (prepared output FP32).
        trainer.preprocess_logits_for_metrics = None
        trainer.compute_metrics = None
        predicted = trainer.predict(data)
        z = predicted.predictions
        assert z.shape == (19, 64, 32768) and z.dtype == np.float32
        assert np.array_equal(predicted.label_ids, data.labels.numpy())
        r["checks"].append(
            "native-BF16-full-predict-19row-order-tail-and-GPU-placement"
        )
        r["prediction_digest"] = array_digest(
            dict(predictions=z, labels=predicted.label_ids)
        )
        with torch.no_grad():
            z64 = oracle(input_ids=data.ids, attention_mask=data.mask).logits
        r["GPU_BF16_vs_CPU_FP64"] = comparison(z, z64, data.mask)
        if a.action == "diagnose":
            assert r["GPU_BF16_vs_CPU_FP64"]["violations"] == 2
            assert (
                abs(
                    r["GPU_BF16_vs_CPU_FP64"]["max_absolute_error"]
                    - 0.003654425659018067
                )
                < 1e-12
            )
            r["checks"].append(
                "original-two-outliers-reproduced-at-unchanged-003-003-gate"
            )
        else:
            assert r["GPU_BF16_vs_CPU_FP64"]["violations"] == 0
            r["checks"].append(
                "explicit-FP32-head-passes-original-all-logit-CPU-FP64-gate"
            )
        zcpu = matched_cpu(cpu, data, autocast=True)
        r["GPU_BF16_vs_matched_CPU_BF16"] = comparison(z, zcpu, data.mask)
        r["CPU_BF16_vs_CPU_FP64"] = comparison(zcpu, z64, data.mask)
        # Diagnostic comparisons are recorded, not silently promoted to qualification.
        r["CPU_BF16_at_GPU_outliers"] = [
            dict(**v, CPU_BF16=float(zcpu[v["row"], v["position"], v["vocab"]]))
            for v in r["GPU_BF16_vs_CPU_FP64"]["outliers"]
        ]
        del zcpu
        gc.collect()
        # Probe the exact local GPU/CPU head inputs/outputs for the affected example.
        features = pad_records([data[12], data[13]])
        features.pop("labels")
        taps = {}
        tap_dtypes = {}
        handles = []
        names = [
            "bert",
            "cls.predictions.transform.dense",
            "cls.predictions.transform.transform_act_fn",
            "cls.predictions.transform.LayerNorm",
            "cls.predictions.decoder",
        ]
        modules = dict(base.named_modules())
        cpu_modules = dict(cpu.named_modules())
        names = [n for n in names if n in modules]

        def capture(name):
            def hook(module, inputs, outputs):
                value = outputs.last_hidden_state if name == "bert" else outputs
                taps[name] = value[1, 42].detach().float().cpu()
                tap_dtypes[name] = str(value.dtype)

            return hook

        for name in names:
            handles.append(modules[name].register_forward_hook(capture(name)))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            base(**{k: v.cuda() for k, v in features.items()})
        for h in handles:
            h.remove()
        gpu_taps = taps
        GPU_tap_dtypes = dict(tap_dtypes)
        taps = {}
        tap_dtypes = {}
        handles = []
        for name in names:
            handles.append(cpu_modules[name].register_forward_hook(capture(name)))
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            cpu(**features)
        for h in handles:
            h.remove()
        r["matched_taps"] = {
            name: dict(
                shape=list(gpu_taps[name].shape),
                GPU_dtype=GPU_tap_dtypes[name],
                CPU_dtype=tap_dtypes[name],
                max_abs_GPU_CPU_BF16=float((gpu_taps[name] - taps[name]).abs().max()),
                exact=torch.equal(gpu_taps[name], taps[name]),
            )
            for name in names
        }
        r["checks"].append("matched-CPU-autocast-and-layer-tap-diagnostics-recorded")
        # Compact metrics operate on actual GPU logits; retain full parity contract.
        expected = metrics(
            type(
                "Prediction",
                (),
                dict(
                    predictions=metrics.preprocess(torch.from_numpy(z), data.labels),
                    label_ids=data.labels,
                ),
            )()
        )
        trainer.compute_metrics = metrics
        trainer.preprocess_logits_for_metrics = metrics.preprocess
        compact = trainer.predict(data)
        for key, value in expected.items():
            assert abs(compact.metrics["test_" + key] - value) <= 1e-11 * max(
                1, abs(value)
            )
        assert compact.metrics["test_loss"] == predicted.metrics["test_loss"]
        r["checks"].append("compact-full-token-metric-parity-and-native-loss-unchanged")
        exact(cpu_tree(base.state_dict()), before)
        assert all(p.grad is None for p in base.parameters())
        r["checks"].append("no-model-state-gradient-or-optimizer-mutation")
        if a.action == "fp32-head":
            assert GPU_tap_dtypes["cls.predictions.decoder"] == "torch.float32"
            scopes.close()
            assert "forward" not in base.cls.__dict__ and not hasattr(
                base.cls, "_bc250_fp32_eval_scope"
            )
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                restored = base(**{k: v.cuda() for k, v in features.items()}).logits
            assert float(restored[1, 42, 16960]) == 0.007293701171875
            r["checks"].append(
                "GPU-FP32-head-dtype-and-exact-native-BF16-scope-restoration"
            )
        peers = [None] * 2
        dist.all_gather_object(peers, r["prediction_digest"])
        assert peers[0] == peers[1]
        r["checks"].append("both-ranks-identical-byte-prediction-digest")
        r.update(
            status="passed",
            token_metrics=expected,
            native_loss=predicted.metrics["test_loss"],
        )
        dist.barrier()
    except BaseException as e:
        r.update(status="failed", error=repr(e))
        raise
    finally:
        scopes.close()
        if dist.is_initialized():
            dist.destroy_process_group()
        atomic_json(a.output, r)


if __name__ == "__main__":
    main()
