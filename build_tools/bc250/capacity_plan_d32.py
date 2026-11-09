#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fixed conservative D32 sizing/admission; estimates are not physical-fit proof."""
CONFIG = dict(
    width=768, heads=12, blocks=6, tokens=256, vocab=4096, global_batch=8, micro=1
)
GIB = 1024**3
PARAMETERS = 45_871_104


def admit(mode, host_available, device_free):
    if mode not in {"single", "ddp", "fsdp"}:
        raise ValueError("Unknown fixed measurement mode")
    # The CPU FP64 oracle/audits coexist with GPU state before measurement.
    # This deliberately leaves host and device headroom, not an OOM search.
    if host_available < 10 * GIB or device_free < 4 * GIB:
        raise ValueError("Insufficient conservative D32 host/device headroom")
    world = 1 if mode == "single" else 2
    return dict(
        preset=CONFIG,
        parameters=PARAMETERS,
        mode=mode,
        world=world,
        fp32_full_parameter_bytes=PARAMETERS * 4,
        expected_master_gradient_AdamW_bytes=PARAMETERS
        * 16
        // (world if mode == "fsdp" else 1),
        CPU_FP64_reference_master_gradient_moments_bytes=PARAMETERS * 32,
        minimum_host_available_bytes=10 * GIB,
        minimum_device_free_bytes=4 * GIB,
        host_available_before_bytes=host_available,
        device_free_before_bytes=device_free,
        cgroup_memory_bytes=10 * GIB,
        cgroup_memory_swap_combined_bytes=12 * GIB,
        estimates_exclude_activations_buckets_temporaries_runtime=True,
        measured_larger_fit=False,
    )
