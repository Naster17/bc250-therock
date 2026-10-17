#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Synthetic CLI math/tiled fresh+resume and old-math fingerprint compatibility."""
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
from validate_qwen35_training_cli import identical

PROJECT = Path("/home/nik/finetune-n7")
WS = Path("/home/nik/bc250work/rocm-10.2-gfx1013")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root = args.output.parent / "synthetic-cli"
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
        for s in ["A small green lamp is on.", "The quiet test ends with seven."]
    ]
    (data / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (data / "val.jsonl").write_text(json.dumps(records[0]) + "\n")
    overlay = str(PROJECT / "runtime/transformers-5.19.0-cp312")
    env = dict(os.environ)
    env["PYTHONPATH"] = overlay + ":" + str(PROJECT) + ":" + env.get("PYTHONPATH", "")
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
    sys.path[:0] = [overlay, str(PROJECT)]
    import replica.train as current

    spec = importlib.util.spec_from_file_location(
        "replica.a07_before", WS / "recovery/a06-a07-before/train.py"
    )
    before = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(before)
    fingerprints = []
    for module, extra in [
        (before, []),
        (current, ["--attention-backend", "math"]),
        (current, ["--attention-backend", "tiled"]),
    ]:
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            module.main(
                common + ["--output", str(root / "preflight"), "--preflight"] + extra
            )
        decoded = json.JSONDecoder().raw_decode(stream.getvalue().lstrip())[0]
        fingerprints.append(decoded["fingerprint"])
    assert fingerprints[0] == fingerprints[1] and fingerprints[1] != fingerprints[2]
    report = dict(
        status="running",
        results=[],
        math_fingerprint_preserved=True,
        backend_fingerprints_distinct=True,
    )
    try:
        for backend in ("math", "tiled"):
            target = root / backend
            target.mkdir()
            for mode in ("fresh", "resumed"):
                cmd = [
                    sys.executable,
                    "-m",
                    "replica.train",
                    *common,
                    "--attention-backend",
                    backend,
                    "--output",
                    str(target / mode),
                ]
                if mode == "resumed":
                    cmd += ["--resume", str(target / "fresh/checkpoint-00000001")]
                log = target / (mode + ".log")
                with log.open("w") as output:
                    code = subprocess.call(
                        cmd,
                        env=env,
                        cwd=PROJECT,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        timeout=110,
                    )
                assert code == 0, log.read_text()[-3000:]
                assert "precision-safe GPU gated-delta active" in log.read_text()
                for bad in [
                    "memory allocation failed with OOM",
                    "MIOpen:",
                    "`causal_conv1d_fn` is falling back",
                ]:
                    assert bad not in log.read_text(), bad
            a, b = [
                target / name / "checkpoint-00000002" for name in ("fresh", "resumed")
            ]
            wa, wb = load_file(a / "adapter_model.safetensors"), load_file(
                b / "adapter_model.safetensors"
            )
            assert identical(wa, wb)
            sa, sb = [
                torch.load(p / "training.pt", map_location="cpu", weights_only=False)
                for p in (a, b)
            ]
            assert identical(sa, sb)
            assert (a / "run.json").read_bytes() == (b / "run.json").read_bytes()
            report["results"].append(
                dict(
                    status="passed",
                    backend=backend,
                    exact_adapter_tensors=len(wa),
                    exact_optimizer_scheduler_rng_progress=True,
                )
            )
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        report["status"] = "passed"
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
