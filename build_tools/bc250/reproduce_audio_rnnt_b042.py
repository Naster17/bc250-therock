#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Minimal first native RNNT fixture from the failed candidate gate."""

import os
import torch
import torchaudio

torch.set_num_threads(4)
torch.manual_seed(4042)
if os.environ.get("AUDIO_REPRO_PROFILE", "1") == "1":
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        wave = torch.randn(2, 2, 37, device="cuda", requires_grad=True)
        a = torch.tensor([[1.2, -0.16, 0.03], [0.9, 0.11, -0.02]], device="cuda")
        b = torch.tensor([[0.4, 0.06, -0.03], [0.3, -0.04, 0.02]], device="cuda")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ):
            torchaudio.functional.lfilter(wave, a, b, clamp=False).sum().backward()
            stream.synchronize()
print(
    "CURRENT_STREAM",
    torch.cuda.current_stream().cuda_stream,
    "DEFAULT",
    torch.cuda.default_stream().cuda_stream,
    flush=True,
)
for time in (4, 35):
    stream = torch.cuda.current_stream() if time == 4 else torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = (torch.randn(2, time, 4, 7) * 0.2).log_softmax(-1).cuda().requires_grad_()
        target = torch.tensor([[1, 2, 3], [2, 2, 1]], dtype=torch.int32, device="cuda")
        times = torch.tensor([time, time - 1], dtype=torch.int32, device="cuda")
        lengths = torch.tensor([3, 2], dtype=torch.int32, device="cuda")
        print("RNNT_BEFORE_STREAM", torch.cuda.current_stream().cuda_stream, flush=True)
        y = torchaudio.functional.rnnt_loss(
            x,
            target,
            times,
            lengths,
            blank=0,
            reduction="none",
            fused_log_softmax=False,
        )
        print("RNNT_AFTER", y, flush=True)
        y.sum().backward()
        print("RNNT_BACKWARD", x.grad.norm(), flush=True)
