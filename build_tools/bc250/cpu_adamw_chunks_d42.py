#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Elementwise first-step FP64 AdamW reference in bounded, independently checked chunks."""
import torch
from distributed_d32_capacity import optimizer

CHUNK_ELEMENTS = 1024**2


def step(initial, gradient):
    assert initial.device.type == gradient.device.type == "cpu"
    ref = torch.nn.Parameter(initial.double())
    ref.grad = gradient.double()
    opt = optimizer(torch.nn.ParameterList([ref]))
    opt.step()
    return ref, opt.state[ref]


def equivalence_check():
    size = 2 * CHUNK_ELEMENTS + 13
    initial = torch.arange(size, dtype=torch.float32) / size * 0.01
    gradient = (torch.arange(size, dtype=torch.float32) % 7 - 3) * 0.001
    full, state = step(initial, gradient)
    chunks = 0
    for start in range(0, size, CHUNK_ELEMENTS):
        stop = min(size, start + CHUNK_ELEMENTS)
        ref, part = step(initial[start:stop], gradient[start:stop])
        assert torch.equal(
            ref.detach().view(torch.uint8), full.detach()[start:stop].view(torch.uint8)
        )
        for key in ["exp_avg", "exp_avg_sq"]:
            assert torch.equal(
                part[key].view(torch.uint8), state[key][start:stop].view(torch.uint8)
            )
        assert int(part["step"]) == int(state["step"]) == 1
        chunks += 1
        del ref, part
    return dict(
        chunk_elements=CHUNK_ELEMENTS,
        synthetic_elements=size,
        chunks=chunks,
        ALL_values_moments_step_exact=True,
    )
