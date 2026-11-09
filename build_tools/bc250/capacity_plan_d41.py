#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Explicit CPU-only oracle budgets; no GPU or scaled-fit admission."""
from capacity_plan_d35 import CONFIG as CONTROL

GIB = 1024**3
CONFIG = dict(CONTROL, width=2048, heads=32, blocks=20)
PARAMETERS = 1024471040
HARD_MEMORY = 11 * GIB
SOFT_MEMORY = 10 * GIB
HOST_RESERVE = 3 * GIB
DISK_RESERVE = 5 * GIB
METADATA_RESERVE = 64 * 1024**2


def parameters(config):
    w = config["width"]
    return (
        config["blocks"] * (12 * w * w + 13 * w)
        + config["vocab"] * w
        + config["positions"] * w
        + 2 * w
    )


def admit(shape, available, free):
    if shape not in {"control", "large"}:
        raise ValueError("Unqualified oracle shape")
    config = CONTROL if shape == "control" else CONFIG
    count = parameters(config)
    assert count == (352957440 if shape == "control" else PARAMETERS)
    largest = max(config["vocab"] * config["width"], 4 * config["width"] ** 2) * 8
    # Conservative CPU plan: resident FP64 params + runtime/agent allowance +
    # checkpoint/engine allowance + three largest per-tensor transient buffers.
    planned = count * 8 + 5 * GIB // 4 + GIB // 2 + 3 * largest
    payload = count * 12 if shape == "large" else largest
    if (
        available < 12 * GIB
        or planned > SOFT_MEMORY
        or available - planned < HOST_RESERVE
        or free < payload + DISK_RESERVE + METADATA_RESERVE
    ):
        raise ValueError("CPU oracle RAM/storage headroom admission refused")
    return dict(
        shape=shape,
        parameters=count,
        largest_FP64_tensor_bytes=largest,
        planned_memory_bytes=planned,
        soft_memory_bytes=SOFT_MEMORY,
        hard_memory_bytes=HARD_MEMORY,
        host_reserve_bytes=HOST_RESERVE,
        available_before_bytes=available,
        free_before_bytes=free,
        required_payload_bytes=payload,
        disk_reserve_bytes=DISK_RESERVE,
        metadata_reserve_bytes=METADATA_RESERVE,
        GPU_admitted=False,
    )


def headroom(memory, available, free):
    if memory >= SOFT_MEMORY or available < HOST_RESERVE or free < DISK_RESERVE:
        raise MemoryError("Owned CPU oracle stopped before exhausting headroom")
