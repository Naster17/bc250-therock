#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Short tiled CLI replay, fingerprint compatibility and backend refusal gate."""

import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from safetensors.torch import load_file
from validate_qwen35_training_cli import PROJECT, identical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.scratch_root / "attention"
    root.mkdir()
    data = root / "data"
    data.mkdir()
    records = [
        dict(
            messages=[
                dict(role="user", content="Reply with a short synthetic sentence."),
                dict(role="assistant", content=s),
            ]
        )
        for s in ("A small green lamp is on.", "The quiet test ends with seven.")
    ]
    (data / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (data / "val.jsonl").write_text(json.dumps(records[0]) + "\n")
    application = args.scratch_root / "application"
    overlay = PROJECT / "runtime/transformers-5.19.0-cp312"
    env = dict(os.environ)
    env.update(
        PYTHONPATH=f"{overlay}:{application}:" + env.get("PYTHONPATH", ""),
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_XET="1",
        REPLICA_LOG_FORMAT="json",
    )
    common = [
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
    sys.path[:0] = [str(overlay), str(application)]
    import replica.train as candidate
    from replica import bc250_checkpoint_provenance as provenance

    spec = importlib.util.spec_from_file_location(
        "replica.n39_entry", args.scratch_root / "entry_train.py"
    )
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    fingerprints = []
    for module, extra in (
        (entry, []),
        (candidate, ["--attention-backend", "math"]),
        (candidate, ["--attention-backend", "tiled"]),
    ):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            module.main(
                common + ["--output", str(root / "preflight"), "--preflight"] + extra
            )
        fingerprints.append(
            json.JSONDecoder().raw_decode(stream.getvalue().lstrip())[0]["fingerprint"]
        )
    assert fingerprints[0] == fingerprints[1] and fingerprints[1] != fingerprints[2]
    report = dict(
        status="running",
        checks=["math-fingerprint-preserved-tiled-distinct"],
        fingerprints=fingerprints,
        commands=[],
    )
    try:
        for mode in ("fresh", "resumed"):
            command = [
                sys.executable,
                "-m",
                "replica.train",
                *common,
                "--attention-backend",
                "tiled",
                "--output",
                str(root / mode),
            ]
            if mode == "resumed":
                command += ["--resume", str(root / "fresh/checkpoint-00000001")]
            report["commands"].append(command)
            with (root / (mode + ".log")).open("w") as log:
                result = subprocess.run(
                    command,
                    env=env,
                    cwd=application,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=150,
                )
            log = (root / (mode + ".log")).read_text()
            assert result.returncode == 0, log[-5000:]
            for bad in (
                "Traceback",
                "MIOpen:",
                "memory allocation failed with OOM",
                "`causal_conv1d_fn` is falling back",
            ):
                assert bad not in log, bad
            events = [
                json.loads(line)
                for line in log.splitlines()
                if line.startswith('{"step":')
            ]
            steps = [e for e in events if "input_tokens" in e]
            assert [e["step"] for e in steps] == ([1, 2] if mode == "fresh" else [2])
            assert all(e["peak_allocated_bytes"] > 0 for e in steps)
        a, b = [root / mode / "checkpoint-00000002" for mode in ("fresh", "resumed")]
        weights = [load_file(p / "adapter_model.safetensors") for p in (a, b)]
        states = [
            torch.load(p / "training.pt", map_location="cpu", weights_only=False)
            for p in (a, b)
        ]
        assert len(weights[0]) == 372 and identical(*weights) and identical(*states)
        report["checks"].append("tiled-gpu-cli-exact-372-adapters-full-state")
        for name in ("run.json", provenance.FILENAME):
            assert (a / name).read_bytes() == (b / name).read_bytes()
        run = json.loads((a / "run.json").read_text())
        record = provenance.read_checkpoint(a, run, run["fingerprint"])
        assert (
            record is not None
            and states[0][provenance.MARKER] == run[provenance.MARKER]
        )
        assert run["config"]["attention_backend"] == "tiled"
        report["checks"].append("tiled-checkpoint-bound-provenance")
        output = root / "refused-math"
        command = [
            sys.executable,
            "-m",
            "replica.train",
            *common,
            "--attention-backend",
            "math",
            "--resume",
            str(root / "fresh/checkpoint-00000001"),
            "--output",
            str(output),
            "--preflight",
        ]
        with (root / "backend-refusal.log").open("w") as log:
            result = subprocess.run(
                command,
                env=env,
                cwd=application,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=60,
            )
        assert (
            result.returncode == 2
            and "fingerprint mismatch" in (root / "backend-refusal.log").read_text()
            and not output.exists()
        )
        report["checks"].append("changed-attention-backend-refused-before-weights")
        report["status"] = "passed"
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
