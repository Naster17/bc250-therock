#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fixed D34 candidate admission, not extrapolated physical-fit certification."""
CONFIG = dict(
    width=1024,
    heads=16,
    blocks=12,
    tokens=128,
    positions=256,
    vocab=8192,
    global_batch=2,
    micro=1,
)
PARAMETERS = 159_807_488
GIB = 1024**3


def admit(mode, host_available, device_free):
    if mode not in {"single", "fsdp"}:
        raise ValueError("Unqualified D34 mode")
    if host_available < 10 * GIB or device_free < 6 * GIB:
        raise ValueError("Insufficient conservative D34 host/device headroom")
    return dict(
        config=CONFIG,
        parameters=PARAMETERS,
        mode=mode,
        host_available_before_bytes=host_available,
        device_free_before_bytes=device_free,
        minimum_host_available_bytes=10 * GIB,
        minimum_device_free_bytes=6 * GIB,
        fp32_state_floor_bytes=PARAMETERS * 16 // (2 if mode == "fsdp" else 1),
        CPU_FP64_oracle_parameters_and_gradients_bytes=PARAMETERS * 16,
        full_CPU_AdamW_state_retained=False,
        estimates_exclude_activations_gathers_allocator_runtime=True,
        larger_than_single_fit=False,
    )
