#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare full and compact native Trainer prediction gathering on two hosts."""
import argparse
import copy
import math
import time
import gc
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from transformers import (
    BertConfig,
    BertForMaskedLM,
    GPT2Config,
    GPT2LMHeadModel,
    TrainingArguments,
)
from transformers.pytorch_utils import Conv1D
import transformers
import accelerate
import bc250_distributed_identity
from bc250_trainer import masked_token_loss
from bc250_trainer_causal import CausalLabelCollator, causal_token_loss
from bc250_trainer_clipping import BC250ClippedTrainer
from bc250_token_metrics import TokenWeightedMetrics
from check_token_metrics_d17 import manual_metrics
from check_token_metrics_d18 import contract_checks
from bc250_compact_token_metrics import CompactTokenMetrics
from distributed_d05_train import atomic_json
from distributed_d10_checkpoint import digest
from distributed_d10_restart import cpu_tree, exact, fingerprint


class OwnedData(torch.utils.data.Dataset):
    def __init__(self, unlabelled=False):
        self.ids = (torch.arange(19 * 12).reshape(19, 12) * 7 + 3) % 96 + 1
        self.ids[:, 0] = torch.arange(19) + 1
        self.mask = torch.zeros_like(self.ids)
        self.labels = self.ids.clone()
        for i in range(19):
            length = 5 + i % 8
            self.mask[i, :length] = 1
            self.ids[i, length:] = 0
            self.labels[i, length:] = -100
            self.labels[i, : i % 3] = -100
        self.labels[:2] = -100
        self.unlabelled = unlabelled

    def __len__(self):
        return 19

    def __getitem__(self, i):
        length = int(self.mask[i].sum())
        value = dict(
            input_ids=self.ids[i, :length], attention_mask=self.mask[i, :length]
        )
        if not self.unlabelled:
            value["labels"] = self.labels[i, :length]
        return value


def pad_records(features):
    size = len(features)
    length = max(len(v["input_ids"]) for v in features)
    result = dict(
        input_ids=torch.zeros((size, length), dtype=torch.long),
        attention_mask=torch.zeros((size, length), dtype=torch.long),
    )
    if "labels" in features[0]:
        result["labels"] = torch.full((size, length), -100, dtype=torch.long)
    for i, value in enumerate(features):
        for key, tensor in value.items():
            result[key][i, : len(tensor)] = tensor
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--case",
        choices=[
            f"{m}-{d}" for m in ["bert", "gpt2"] for d in ["fp32", "bf16", "fp16"]
        ],
        required=True,
    )
    p.add_argument("--action", choices=["eval"], required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = dict(status="running", case=a.case, checks=[], CPU_contract_checks=[])
    try:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        r["CPU_contract_checks"] = contract_checks()
        model_name, precision = a.case.split("-")
        causal = model_name == "gpt2"
        dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
            precision
        ]
        torch.manual_seed(773)
        if causal:
            cfg = GPT2Config(
                vocab_size=97,
                n_positions=32,
                n_ctx=32,
                n_embd=32,
                n_layer=2,
                n_head=4,
                n_inner=64,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
                use_cache=False,
                attn_implementation="eager",
            )
            base = GPT2LMHeadModel(cfg)
        else:
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
        oracle = copy.deepcopy(base).double().eval()
        types = []
        for layer in base.modules():
            if isinstance(layer, (torch.nn.Linear, Conv1D)):
                layer.register_forward_hook(lambda m, x, y: types.append(str(y.dtype)))
        data = OwnedData()
        expected_labels = data.labels.clone()
        if causal:
            expected_labels[:, 0] = -100
        collator = CausalLabelCollator(pad_records) if causal else pad_records
        metrics_callback = TokenWeightedMetrics(causal=causal, chunk_tokens=7)
        args = TrainingArguments(
            output_dir=str(a.output.parent / "trainer-output"),
            per_device_eval_batch_size=2,
            bf16=precision == "bf16",
            fp16=precision == "fp16",
            max_grad_norm=0.05,
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
            data_collator=collator,
            compute_loss_func=causal_token_loss if causal else masked_token_loss,
            compute_metrics=metrics_callback,
        )
        acc = trainer.accelerator
        rank = acc.process_index
        assert (
            acc.num_processes == 2
            and acc.distributed_type.value == "MULTI_GPU"
            and acc.state.backend == "nccl"
        )
        assert all(v.device.type == "cuda" for v in base.parameters())
        assert torch.cuda.get_device_properties(0).gcnArchName == "gfx1013:xnack-"
        tool = Path(bc250_distributed_identity.__file__).parent
        package = Path(transformers.__file__).parent
        config = dict(
            case=a.case,
            world=2,
            records=19,
            vocab=97,
            variable_sequence="5-12",
            per_device_eval_batch=2,
            workers=0,
            precision=precision,
            causal=causal,
            attention="eager",
            dropout=0,
            use_cache=False,
            transformers=transformers.__version__,
            accelerate=accelerate.__version__,
            fixture=digest(Path(__file__)),
            native=digest(tool / "torch-identity.json"),
            trainer=digest(package / "trainer.py"),
            helpers={
                name: digest(Path(__file__).with_name(name))
                for name in [
                    "bc250_trainer.py",
                    "bc250_trainer_finite.py",
                    "bc250_trainer_clipping.py",
                    "bc250_trainer_causal.py",
                    "bc250_token_metrics.py",
                    "check_token_metrics_d17.py",
                    "check_token_metrics_d18.py",
                    "bc250_compact_token_metrics.py",
                ]
            },
        )
        before = cpu_tree(base.state_dict())
        r.update(rank=rank, config=config, source_sha256=config["fixture"])
        payloads = []
        original_gather = acc.gather_for_metrics

        def recorded_gather(value, *positional, **keywords):
            if isinstance(value, torch.Tensor):
                payloads.append(
                    dict(
                        shape=list(value.shape),
                        dtype=str(value.dtype),
                        bytes=value.numel() * value.element_size(),
                    )
                )
            return original_gather(value, *positional, **keywords)

        acc.gather_for_metrics = recorded_gather
        # Warm model/collectives before the bounded ABBA comparison.
        trainer.predict(data)
        compact = CompactTokenMetrics(causal=causal, chunk_tokens=7)
        samples = []
        full_logits = None
        full_labels = None
        baseline = None
        for mode in ["full", "compact", "compact", "full"]:
            trainer.compute_metrics = metrics_callback if mode == "full" else compact
            trainer.preprocess_logits_for_metrics = (
                None if mode == "full" else compact.preprocess
            )
            payloads.clear()
            trainer.gather_function = recorded_gather
            gc.collect()
            dist.barrier()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            predicted = trainer.predict(data, metric_key_prefix="predict")
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            metric = {
                k: predicted.metrics["predict_" + k]
                for k in [
                    "token_nll",
                    "token_perplexity",
                    "token_accuracy",
                    "active_tokens",
                    "evaluated_examples",
                ]
            }
            assert predicted.label_ids.shape == (19, 12)
            assert np.array_equal(predicted.label_ids, expected_labels.numpy())
            if mode == "full":
                assert (
                    predicted.predictions.shape == (19, 12, 97)
                    and predicted.predictions.dtype == np.float32
                )
                if full_logits is None:
                    full_logits = predicted.predictions.copy()
                    full_labels = predicted.label_ids.copy()
                    baseline = metric
                else:
                    assert np.array_equal(full_logits, predicted.predictions)
            else:
                assert (
                    predicted.predictions.shape == (19, 3)
                    and predicted.predictions.dtype == np.float64
                )
            for key, value in baseline.items():
                assert math.isclose(metric[key], value, rel_tol=1e-11, abs_tol=1e-11)
            prediction_payload = [
                v
                for v in payloads
                if (
                    len(v["shape"]) == 3
                    if mode == "full"
                    else v["shape"][-1:] == [3] and v["dtype"] == "torch.float64"
                )
            ]
            assert len(prediction_payload) == 5, payloads
            samples.append(
                dict(
                    mode=mode,
                    elapsed_s=elapsed,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    prediction_array_bytes=predicted.predictions.nbytes,
                    label_array_bytes=predicted.label_ids.nbytes,
                    prediction_gather_input_bytes=sum(
                        v["bytes"] for v in prediction_payload
                    ),
                    gather_inputs=list(payloads),
                    metrics=metric,
                    native_loss=predicted.metrics["predict_loss"],
                )
            )
        assert types and set(types) == {str(dtype)}
        r["checks"].append(
            "ABBA-full-compact-exact19-rows-tail-dedup-label-order-and-metric-parity"
        )
        r["checks"].append("five-pre-gather-prediction-payloads-per-mode-observed")
        r["checks"].append("GPU-parameter-and-autocast-dtype-placement")
        with torch.no_grad():
            reference_logits = oracle(
                input_ids=data.ids, attention_mask=data.mask
            ).logits
        active_input = data.mask.bool()
        atol = {"fp32": 3e-6, "bf16": 0.003, "fp16": 0.001}[precision]
        rtol = {"fp32": 5e-5, "bf16": 0.003, "fp16": 0.001}[precision]
        torch.testing.assert_close(
            torch.from_numpy(full_logits)[active_input].double(),
            reference_logits[active_input],
            atol=atol,
            rtol=rtol,
        )
        expected = manual_metrics(full_logits, full_labels, causal)
        high_precision = manual_metrics(
            reference_logits.numpy(), expected_labels.numpy(), causal
        )
        assert (
            expected["active_tokens"] == (120 if causal else 126)
            and expected["evaluated_examples"] == 19
        )
        for key, value in expected.items():
            assert math.isclose(baseline[key], value, rel_tol=1e-11, abs_tol=1e-11)
        assert math.isclose(
            expected["token_nll"],
            high_precision["token_nll"],
            rel_tol=rtol,
            abs_tol=atol,
        )
        r["checks"].append(
            "full-GPU-logits-and-compact-NLL-vs-CPU-FP64-model-and-manual-NumPy"
        )
        trainer.compute_metrics = compact
        trainer.preprocess_logits_for_metrics = compact.preprocess
        evaluated = trainer.evaluate(metric_key_prefix="eval")
        for key, value in baseline.items():
            assert math.isclose(
                evaluated["eval_" + key], value, rel_tol=1e-11, abs_tol=1e-11
            )
        for sample in samples:
            assert sample["native_loss"] == evaluated["eval_loss"]
        difference = abs(evaluated["eval_loss"] - expected["token_nll"])
        assert difference > 1e-6
        r["checks"].append("compact-evaluate-parity-and-native-loss-unchanged")
        r["ABBA_samples"] = samples
        r["checks"].append("compact-prediction-array456B-vs-full88464B")
        assert all(
            s["prediction_array_bytes"] == (88464 if s["mode"] == "full" else 456)
            for s in samples
        )
        # Label-free inference must bypass the training collator's required labels.
        trainer.compute_metrics = metrics_callback
        trainer.preprocess_logits_for_metrics = None
        trainer.data_collator = pad_records
        unlabelled = trainer.predict(
            OwnedData(unlabelled=True), metric_key_prefix="unlabelled"
        )
        trainer.data_collator = collator
        assert unlabelled.label_ids is None and unlabelled.predictions.shape == (
            19,
            12,
            97,
        )
        assert np.array_equal(full_logits, unlabelled.predictions)
        assert not any(
            key.endswith("_loss") or "token_nll" in key for key in unlabelled.metrics
        )
        r["checks"].append(
            "label-free-predict-exact-same-logits-without-label-loss-or-token-metrics"
        )
        exact(cpu_tree(base.state_dict()), before)
        assert all(v.grad is None for v in base.parameters())
        r["checks"].append("native-evaluation-no-parameter-or-gradient-mutation")
        peers = [None] * 2
        dist.all_gather_object(peers, expected)
        assert peers[0] == peers[1]
        r["checks"].append("both-ranks-identical-global-token-metrics")
        r.update(
            status="passed",
            token_metrics=expected,
            CPU_FP64_token_nll=high_precision["token_nll"],
            native_eval_loss=evaluated["eval_loss"],
            native_loss_token_NLL_gap=difference,
            prediction_digest=fingerprint(
                dict(
                    predictions=full_logits.tolist(),
                    labels=full_labels.tolist(),
                )
            ),
            evaluated_metric_keys=sorted(evaluated),
            label_free_metric_keys=sorted(unlabelled.metrics),
        )
        dist.barrier()
    except BaseException as error:
        r.update(status="failed", error=repr(error))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        atomic_json(a.output, r)


if __name__ == "__main__":
    main()
