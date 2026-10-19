#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Installed trainer replay against the identical candidate checkpoint/control."""

import argparse
import json
import os
from pathlib import Path
import subprocess

import torch
from safetensors.torch import load_file
from validate_qwen35_training_cli import PROJECT, identical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.scratch_root
    torch.set_num_threads(4)
    for name in ("train.py", "bc250_checkpoint_provenance.py"):
        assert (PROJECT / "replica" / name).read_bytes() == (
            root / "application/replica" / name
        ).read_bytes()
    report = dict(status="running", checks=[])
    try:
        fresh = json.loads((root / "fresh-result.json").read_text())
        command = fresh["command"][:]
        command[command.index("--output") + 1] = str(root / "installed")
        command += ["--resume", str(root / "fresh/checkpoint-00000001")]
        env = dict(os.environ)
        env.update(
            PYTHONPATH=str(PROJECT / "runtime/transformers-5.19.0-cp312")
            + ":"
            + str(PROJECT)
            + ":"
            + env.get("PYTHONPATH", ""),
            OMP_NUM_THREADS="4",
            MKL_NUM_THREADS="4",
            HF_HUB_OFFLINE="1",
            HF_HUB_DISABLE_XET="1",
            REPLICA_LOG_FORMAT="json",
        )
        with (root / "installed.log").open("w") as log:
            result = subprocess.run(
                command,
                cwd=PROJECT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=460,
            )
        log = (root / "installed.log").read_text()
        assert result.returncode == 0, log[-5000:]
        assert "Legacy checkpoint has no runtime provenance" not in log
        events = [
            json.loads(line) for line in log.splitlines() if line.startswith('{"step":')
        ]
        step = next(e for e in events if "input_tokens" in e)
        assert (
            step["step"] == 2
            and step["input_tokens"] == 16384
            and step["peak_allocated_bytes"] > 0
        )
        for key in ("loss", "target_tokens", "input_tokens", "lr", "position"):
            assert step[key] == fresh["steps"][-1][key], key
        report["checks"].append(dict(name="installed_gpu_cli_exact_loss", step=step))
        control = root / "fresh/checkpoint-00000002"
        resumed = root / "installed/checkpoint-00000002"
        wa, wb = [
            load_file(p / "adapter_model.safetensors") for p in (control, resumed)
        ]
        assert len(wa) == 372 and identical(wa, wb)
        sa, sb = [
            torch.load(p / "training.pt", map_location="cpu", weights_only=False)
            for p in (control, resumed)
        ]
        assert identical(sa, sb)
        report["checks"].append(dict(name="installed_exact_372_adapters_full_state"))
        for name in ("run.json", "runtime-provenance.json"):
            assert (control / name).read_bytes() == (resumed / name).read_bytes()
        report["checks"].append(dict(name="installed_bound_provenance_identical"))
        previous = Path("/tmp/bc250-training-n36-r1/fresh/checkpoint-00000002")
        assert identical(wa, load_file(previous / "adapter_model.safetensors"))
        previous_state = torch.load(
            previous / "training.pt", map_location="cpu", weights_only=False
        )
        numerical_state = dict(sa)
        numerical_state.pop("runtime_provenance_sha256")
        assert identical(
            numerical_state, previous_state
        ), "Numerical state changed relative to pre-N3.7 control"
        report["checks"].append(
            dict(name="pre_n37_numerical_state_exact", adapter_tensors=372)
        )
        report["status"] = "passed"
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
