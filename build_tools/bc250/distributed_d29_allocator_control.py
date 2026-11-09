#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Same D29 reference fixture with checkpointing off, for bounded allocator control."""
from types import SimpleNamespace
import torch
import distributed_d29_trainer as fixture
from distributed_d25_trainer import main


def configure_args(args):
    assert not args.gradient_checkpointing


def finalize(result):
    expected = 7 if len(result["transcript"]) == 4 else 3
    assert [c[0] for c in fixture.counts] == [expected, expected], fixture.counts
    result["checkpoint_block_calls"] = [c[0] for c in fixture.counts]
    result["allocator_peak_bytes"] = torch.cuda.max_memory_allocated()
    result["allocator_reserved_peak_bytes"] = torch.cuda.max_memory_reserved()
    result["checks"].append("GPU-both-blocks-forward-without-recomputation-control")


if __name__ == "__main__":
    control = SimpleNamespace(**vars(fixture))
    control.__file__ = __file__
    control.name = "D29 FP32 PEFT checkpointing-off allocator reference control"
    control.trainer_class = fixture.previous.BC250PeftFSDPTrainer
    control.configure_args = configure_args
    control.finalize = finalize
    main(fixture=control)
