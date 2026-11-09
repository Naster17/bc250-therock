#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Matched BF16 checkpointing-off allocator control for the D31 tiny harness."""
from types import SimpleNamespace
import torch
import distributed_d31_trainer as fixture
from distributed_d25_trainer import main


def configure_args(args):
    assert args.bf16 and not args.fp16 and not args.gradient_checkpointing


def finalize(result):
    expected = 7 if len(result["transcript"]) == 4 else 3
    assert [c[0] for c in fixture.recompute.counts] == [expected, expected]
    result["checkpoint_block_calls"] = [c[0] for c in fixture.recompute.counts]
    result["allocator_peak_bytes"] = torch.cuda.max_memory_allocated()
    result["allocator_reserved_peak_bytes"] = torch.cuda.max_memory_reserved()
    result["checks"].append("GPU-BF16-block-forward-without-recomputation-control")


if __name__ == "__main__":
    control = SimpleNamespace(**vars(fixture))
    control.__file__ = __file__
    control.name = "D31 matched BF16 checkpointing-off allocator reference control"
    control.trainer_class = fixture.previous.BC250BF16PeftFSDPTrainer
    control.configure_args = configure_args
    control.finalize = finalize
    main(fixture=control)
