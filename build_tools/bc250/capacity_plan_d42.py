#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Source-bound oracle identity and explicit GPU audit admission; never single OOM."""
import json
from pathlib import Path
from artifact_d35 import digest, source_hashes
from artifact_d41 import sources as oracle_sources
from capacity_plan_d41 import CONFIG, CONTROL, parameters, GIB

DESCRIPTOR_SHA = "a0af0b1f62295742d23b8e809ac320e3da60d0d8fdc2e4783d9fa6c1e068a867"
HOST_REGISTRY_SHA256 = {
    0: "d5841f6d6e2f76265c4055acc81ab4a9fcbeb9cd3a6739205469ba0509862866",
    1: "d5212c7b7d351577cd029121d6cc0f9ded01d0a7da0e136716c643ddcff54327",
}
READINESS_SECONDS = 600
CHILD_SECONDS = 1400
GUARD_SECONDS = 1800
HOST_RESERVE = 3 * GIB
DISK_RESERVE = 5 * GIB


def descriptor(shape):
    if shape not in {"control", "large"}:
        raise ValueError("Unknown audit shape")
    path = Path(__file__).with_name("oracle_descriptor_d42.json")
    if digest(path) != DESCRIPTOR_SHA:
        raise ValueError("CPU oracle descriptor changed")
    record = json.loads(path.read_text())[shape]
    config = CONTROL if shape == "control" else CONFIG
    manifest, worker, final, state, ack = [
        record[k] for k in ["manifest", "worker", "final", "state", "ack"]
    ]
    assert manifest["config"] == config and manifest["parameters"] == parameters(config)
    assert manifest["sources"] == (
        source_hashes() if shape == "control" else oracle_sources()
    )
    assert (
        manifest["status"] == worker["status"] == final["status"] == "passed"
        and final["worker"] == worker
    )
    assert final["registry_unchanged"] and final["kernel_log_unchanged"]
    assert state["status"] == "acknowledged" and all(
        state[k] == ack[k] for k in ["nonce", "boot_id", "phase"]
    )
    assert record["manifest_sha256"] == worker["manifest_sha256"]
    if shape == "large":
        assert record["finite_blob_scan"] == dict(
            initial_values=parameters(config),
            gradient_values=parameters(config),
            max_chunk_bytes=1024**2,
        )
    return record


def admit(shape, available, free, *, device_free=None):
    config = CONTROL if shape == "control" else CONFIG if shape == "large" else None
    if config is None:
        raise ValueError("Unknown audit shape")
    count = parameters(config)
    # Resident FP32 params/gradients/two AdamW moments, each sharded over2 ranks.
    state = count * 8
    incremental = state + GIB  # bounded gather/activation overlap before model creation
    planned = (
        state + 3 * GIB
    )  # guardian also budgets resident runtime and CPU audit buffers
    if free < DISK_RESERVE + 64 * 1024**2:
        raise ValueError("Audit disk reserve refused")
    wanted = planned if device_free is None else incremental
    if available < wanted + HOST_RESERVE:
        raise ValueError("Audit host RAM admission refused")
    if device_free is not None and device_free < incremental:
        raise ValueError("Audit device RAM admission refused")
    return dict(
        shape=shape,
        parameters=count,
        local_FP32_state_floor_bytes=state,
        planned_host_bytes=wanted,
        available_before_bytes=available,
        free_disk_before_bytes=free,
        disk_reserve_bytes=DISK_RESERVE,
        device_free_before_bytes=device_free,
        host_reserve_bytes=HOST_RESERVE,
        single_resident_FP32_AdamW_floor_bytes=count * 16,
    )


def headroom(memory, available, free):
    if memory >= 9 * GIB or available < HOST_RESERVE or free < DISK_RESERVE:
        raise MemoryError("Owned D42 audit headroom stop")
