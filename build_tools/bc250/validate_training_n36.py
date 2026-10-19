#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Owned 1024-token/rank16/accumulation16 deployed trainer CLI replay."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from safetensors.torch import load_file
from validate_qwen35_training_cli import PROJECT, identical


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("fresh", "resumed"), required=True)
    args = parser.parse_args()
    root = args.scratch_root
    assert root.parent == Path("/tmp") and root.is_dir()
    torch.set_num_threads(4)
    preset = Path("/workspace/rocblas-gemm-overrides.csv")
    assert os.environ.get("ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH") == str(preset)
    assert "LD_PRELOAD" not in os.environ
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(PROJECT / "runtime/transformers-5.19.0-cp312")
        + ":"
        + str(PROJECT)
        + ":"
        + env.get("PYTHONPATH", ""),
        REPLICA_LOG_FORMAT="json",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_XET="1",
    )
    sys.path[:0] = [str(PROJECT / "runtime/transformers-5.19.0-cp312"), str(PROJECT)]
    from transformers import AutoTokenizer
    from replica.formatting import encode_record

    identity = dict(
        preset_sha256=sha(preset),
        sources={
            name: sha(PROJECT / "replica" / name)
            for name in (
                "train.py",
                "formatting.py",
                "bc250_ops.py",
                "bc250_gated_delta.py",
            )
        },
        test_sha256=sha(Path(__file__)),
    )
    report = dict(status="running", checks=[], mode=args.mode, identity=identity)
    try:
        if args.mode == "fresh":
            data = root / "data"
            data.mkdir()
            tokenizer = AutoTokenizer.from_pretrained(
                PROJECT / "models/Qwen3.5-2B-Base", use_fast=True, local_files_only=True
            )
            records, geometry = [], []
            for index in range(32):
                # Each repeated word is one token here; verify through the actual formatter.
                user_words = 600
                assistant_words = 64 + index * 7
                for _ in range(8):
                    record = dict(
                        messages=[
                            dict(
                                role="user",
                                content=f"Synthetic case {index}."
                                + " blue" * user_words,
                            ),
                            dict(
                                role="assistant",
                                content="green" + " green" * assistant_words,
                            ),
                        ]
                    )
                    encoded = encode_record(record, tokenizer, 2048)
                    difference = 1024 - len(encoded["input_ids"])
                    if not difference:
                        break
                    user_words += difference
                assert len(encoded["input_ids"]) == 1024
                count = sum(x != -100 for x in encoded["labels"][1:])
                assert 0 < count < 1023
                record.update(token_count=1024, supervised_tokens=count)
                records.append(record)
                geometry.append(
                    dict(
                        length=1024,
                        target_tokens=count,
                        ids_sha256=hashlib.sha256(
                            json.dumps(encoded, sort_keys=True).encode()
                        ).hexdigest(),
                    )
                )
            assert len({r["target_tokens"] for r in geometry}) == 32
            (data / "train.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in records)
            )
            (data / "val.jsonl").write_text(json.dumps(records[0]) + "\n")
            (root / "fixture.json").write_text(
                json.dumps(dict(identity=identity, geometry=geometry), indent=2) + "\n"
            )
        fixture = json.loads((root / "fixture.json").read_text())
        assert fixture["identity"] == identity
        report["checks"].append(
            dict(
                name="pinned_masked_1024_fixture", records=32, distinct_target_counts=32
            )
        )
        common = [
            sys.executable,
            "-m",
            "replica.train",
            "--model",
            str(PROJECT / "models/Qwen3.5-2B-Base"),
            "--dataset",
            str(root / "data"),
            "--max-seq-length",
            "1024",
            "--epochs",
            "1",
            "--max-steps",
            "2",
            "--accumulation-steps",
            "16",
            "--learning-rate",
            "0.0001",
            "--warmup-steps",
            "0",
            "--rank",
            "16",
            "--alpha",
            "32",
            "--targets",
            "auto-linear",
            "--save-every",
            "1",
            "--eval-every",
            "1",
            "--schedule",
            "cosine",
            "--seed",
            "31936",
            "--device",
            "cuda",
            "--dtype",
            "bf16",
            "--sdpa",
            "math",
            "--output",
            str(root / args.mode),
        ]
        if args.mode == "resumed":
            common += ["--resume", str(root / "fresh/checkpoint-00000001")]
        report["command"] = common
        logpath = root / (args.mode + ".log")
        with logpath.open("w") as log:
            completed = subprocess.run(
                common,
                env=env,
                cwd=PROJECT,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=460,
            )
        log = logpath.read_text()
        assert completed.returncode == 0, log[-6000:]
        for forbidden in (
            "MIOpen:",
            "memory allocation failed with OOM",
            "Traceback",
            "`causal_conv1d_fn` is falling back",
        ):
            assert forbidden not in log, forbidden
        events = []
        for line in log.splitlines():
            if line.startswith('{"step":'):
                events.append(json.loads(line))
        steps = [e for e in events if "input_tokens" in e]
        assert [e["step"] for e in steps] == ([1, 2] if args.mode == "fresh" else [2])
        assert all(
            e["input_tokens"] == 16384 and e["peak_allocated_bytes"] > 0 for e in steps
        )
        assert all(torch.isfinite(torch.tensor(e["loss"])) for e in steps)
        report["steps"] = steps
        report["checks"].append(
            dict(
                name="actual_gpu_cli_finite_updates",
                steps=len(steps),
                input_tokens_per_step=16384,
            )
        )
        checkpoint = root / args.mode / "checkpoint-00000002"
        run = json.loads((checkpoint / "run.json").read_text())
        config = run["config"]
        assert (
            config["accumulation_steps"] == 16
            and config["rank"] == 16
            and config["total_steps"] == 2
        )
        assert (
            run["fingerprint"]
            == hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        )
        state = torch.load(
            checkpoint / "training.pt", map_location="cpu", weights_only=False
        )
        assert (
            state["step"] == 2
            and state["position"] == 32
            and sorted(state["order"]) == list(range(32))
        )
        assert state["fingerprint"] == run["fingerprint"]
        report["fingerprint"] = run["fingerprint"]
        report["checks"].append(
            dict(name="checkpoint_progress_fingerprint", step=2, position=32)
        )
        if args.mode == "resumed":
            control = root / "fresh/checkpoint-00000002"
            weights_a, weights_b = [
                load_file(p / "adapter_model.safetensors")
                for p in (control, checkpoint)
            ]
            assert len(weights_a) == 372 and identical(
                weights_a, weights_b
            ), "Adapter replay differs"
            assert identical(
                state,
                torch.load(
                    control / "training.pt", map_location="cpu", weights_only=False
                ),
            ), "Full training-state replay differs"
            assert (control / "run.json").read_bytes() == (
                checkpoint / "run.json"
            ).read_bytes()
            prior = json.loads((root / "fresh-result.json").read_text())
            assert identical(
                {
                    k: v
                    for k, v in prior["steps"][-1].items()
                    if k
                    in (
                        "step",
                        "position",
                        "loss",
                        "lr",
                        "input_tokens",
                        "target_tokens",
                    )
                },
                {
                    k: v
                    for k, v in steps[0].items()
                    if k
                    in (
                        "step",
                        "position",
                        "loss",
                        "lr",
                        "input_tokens",
                        "target_tokens",
                    )
                },
            )
            report["checks"].append(
                dict(
                    name="exact_fresh_process_replay",
                    adapter_tensors=372,
                    full_optimizer_scheduler_rng_progress=True,
                    loss=True,
                )
            )
        report["status"] = "passed"
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        (root / (args.mode + "-result.json")).write_text(
            json.dumps(report, indent=2) + "\n"
        )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
