#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fixed D35 candidate admission, not extrapolated physical-fit certification."""
CONFIG = dict(
    width=1536,
    heads=24,
    blocks=12,
    tokens=128,
    positions=256,
    vocab=8192,
    global_batch=2,
    micro=1,
)
PARAMETERS = 352_957_440
GIB = 1024**3


def admit(mode, host_available, device_free):
    if mode not in {"single", "fsdp"}:
        raise ValueError("Unqualified D35 mode")
    if host_available < 11 * GIB or device_free < 9 * GIB:
        raise ValueError("Insufficient conservative D35 host/device headroom")
    return dict(
        config=CONFIG,
        parameters=PARAMETERS,
        mode=mode,
        host_available_before_bytes=host_available,
        device_free_before_bytes=device_free,
        minimum_host_available_bytes=11 * GIB,
        minimum_device_free_bytes=9 * GIB,
        fp32_state_floor_bytes=PARAMETERS * 16 // (2 if mode == "fsdp" else 1),
        CPU_FP64_oracle_parameters_and_gradients_bytes=PARAMETERS * 16,
        full_CPU_AdamW_state_retained=False,
        estimates_exclude_activations_gathers_allocator_runtime=True,
        larger_than_single_fit=False,
    )
