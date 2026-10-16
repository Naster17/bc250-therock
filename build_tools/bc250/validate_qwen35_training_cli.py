#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fresh/resume gate through the deployed user trainer using new synthetic files."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from safetensors.torch import load_file

PROJECT = Path("/home/nik/finetune-n7")
WS = Path("/home/nik/bc250work/rocm-10.2-gfx1013")


def identical(left, right):
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            identical(left[k], right[k]) for k in left
        )
    if isinstance(left, (tuple, list)):
        return (
            type(left) == type(right)
            and len(left) == len(right)
            and all(identical(a, b) for a, b in zip(left, right))
        )
    return left == right


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument(
        "--guard-reference",
        type=Path,
        help="Prior synthetic-cli root for mismatch refusal tests",
    )
    args = ap.parse_args()
    deployed = PROJECT / "replica/bc250_ops.py"
    assert deployed.read_bytes() == (WS / "helpers/qwen35_training_ops.py").read_bytes()
    root = args.output.parent / "synthetic-cli"
    root.mkdir()
    data = root / "data"
    data.mkdir()
    records = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with a short synthetic test sentence.",
                },
                {"role": "assistant", "content": s},
            ]
        }
        for s in ("The small green lamp is on.", "A quiet test ends with seven.")
    ]
    (data / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (data / "val.jsonl").write_text(json.dumps(records[0]) + "\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(PROJECT / "runtime/transformers-5.19.0-cp312")
        + ":"
        + str(PROJECT)
        + ":"
        + env.get("PYTHONPATH", "")
    )
    common = [
        sys.executable,
        "-m",
        "replica.train",
        "--model",
        str(PROJECT / "models/Qwen3.5-2B-Base"),
        "--dataset",
        str(data),
        "--max-seq-length",
        "1024",
        "--epochs",
        "1",
        "--max-steps",
        "2",
        "--accumulation-steps",
        "1",
        "--learning-rate",
        "0.0001",
        "--warmup-steps",
        "0",
        "--rank",
        "4",
        "--alpha",
        "8",
        "--targets",
        "auto-linear",
        "--save-every",
        "1",
        "--eval-every",
        "1",
        "--schedule",
        "cosine",
        "--seed",
        "484",
        "--device",
        "cuda",
        "--dtype",
        "bf16",
        "--sdpa",
        "math",
    ]
    result = {
        "source_sha256": hashlib.sha256(deployed.read_bytes()).hexdigest(),
        "synthetic_root": str(root),
        "jobs": [],
    }
    if args.guard_reference:
        checkpoint = args.guard_reference / "fresh/checkpoint-00000001"
        config = json.loads((checkpoint / "adapter_config.json").read_text())
        try:
            for mode in ("missing-target", "extra-target", "wrong-rank", "wrong-alpha"):
                fixture = root / (mode + "-fixture")
                fixture.mkdir()
                modified = dict(config)
                if mode == "missing-target":
                    modified["target_modules"] = ["q_proj"]
                elif mode == "extra-target":
                    modified["target_modules"] = config["target_modules"] + ["lm_head"]
                elif mode == "wrong-rank":
                    modified["r"] += 1
                else:
                    modified["lora_alpha"] += 1
                for file in checkpoint.iterdir():
                    if file.name != "adapter_config.json":
                        (fixture / file.name).symlink_to(
                            file, target_is_directory=file.is_dir()
                        )
                (fixture / "adapter_config.json").write_text(json.dumps(modified))
                command = common + [
                    "--output",
                    str(root / mode),
                    "--resume",
                    str(fixture),
                ]
                with (root / (mode + ".log")).open("w") as stream:
                    completed = subprocess.run(
                        command,
                        env=env,
                        cwd=PROJECT,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        timeout=90,
                    )
                log = (root / (mode + ".log")).read_text()
                assert (
                    completed.returncode == 2
                    and "Checkpoint adapter target/rank/alpha mismatch" in log
                ), log[-2000:]
                assert not (root / mode).exists(), "Refused resume wrote an output"
                result["jobs"].append(
                    dict(
                        mode=mode,
                        returncode=completed.returncode,
                        correctly_refused=True,
                    )
                )
            result["passed"] = True
        finally:
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        return
    try:
        for mode in ("fresh", "resumed"):
            command = common + ["--output", str(root / mode)]
            if mode == "resumed":
                command += ["--resume", str(root / "fresh/checkpoint-00000001")]
            with (root / (mode + ".log")).open("w") as stream:
                completed = subprocess.run(
                    command,
                    env=env,
                    cwd=PROJECT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=210,
                )
            log = (root / (mode + ".log")).read_text()
            result["jobs"].append(
                {
                    "mode": mode,
                    "returncode": completed.returncode,
                    "log": str(root / (mode + ".log")),
                }
            )
            assert completed.returncode == 0, log[-5000:]
            assert result["source_sha256"] in log, "Deployed installer not executed"
            for forbidden in (
                "MIOpen:",
                "`causal_conv1d_fn` is falling back",
                "memory allocation failed with OOM",
            ):
                assert forbidden not in log, forbidden
            assert (
                "`chunk_gated_delta_rule` is falling back" in log
            ), "Legitimate remaining warning was hidden"
        fresh = root / "fresh/checkpoint-00000002"
        resumed = root / "resumed/checkpoint-00000002"
        weights_a = load_file(fresh / "adapter_model.safetensors")
        weights_b = load_file(resumed / "adapter_model.safetensors")
        assert identical(
            weights_a, weights_b
        ), "Fresh and resumed adapter tensors differ"
        state_a = torch.load(
            fresh / "training.pt", map_location="cpu", weights_only=False
        )
        state_b = torch.load(
            resumed / "training.pt", map_location="cpu", weights_only=False
        )
        assert identical(
            state_a, state_b
        ), "Optimizer/scheduler/RNG/progress continuation differs"
        assert (fresh / "run.json").read_bytes() == (resumed / "run.json").read_bytes()
        result.update(
            passed=True,
            exact_adapter_tensors=len(weights_a),
            exact_training_state=True,
            steps_fresh=2,
            steps_resumed=1,
        )
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
