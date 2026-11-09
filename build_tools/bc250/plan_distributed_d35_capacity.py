#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CPU-only next-capacity blockers; state floors never admit GPU execution."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for host in ["104", "105"]:
        parser.add_argument("--host" + host + "-memtotal-kib", type=int, required=True)
        parser.add_argument("--host" + host + "-free-disk-kib", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    width = 2048
    block = 12 * width * width + 13 * width
    parameters = 20 * block + 8192 * width + 256 * width + 2 * width
    hosts = []
    for host in ["104", "105"]:
        ram = getattr(args, "host" + host + "_memtotal_kib") * 1024
        disk = getattr(args, "host" + host + "_free_disk_kib") * 1024
        hosts.append(
            dict(
                host=host,
                observed_MemTotal_bytes=ram,
                observed_free_disk_bytes=disk,
                resident_single_FP32_AdamW_floor_exceeds_RAM=parameters * 16 > ram,
                unchanged_reference_storage_plus_5GiB_fits=parameters * 12 + 5 * 1024**3
                <= disk,
                unchanged_actual_spool_plus_5GiB_fits=parameters * 4 + 5 * 1024**3
                <= disk,
            )
        )
    result = dict(
        status="planning-only",
        parameters=parameters,
        blocks=20,
        width=width,
        heads=32,
        vocabulary=8192,
        positions=256,
        sequence=128,
        global_batch=2,
        micro=1,
        single_FP32_parameter_gradient_AdamW_floor_bytes=parameters * 16,
        two_rank_FP32_state_floor_bytes_each=parameters * 8,
        unchanged_CPU_FP64_parameter_gradient_floor_bytes=parameters * 16,
        unchanged_reference_payload_bytes_each=parameters * 12,
        unchanged_whole_model_summon_extra_parameter_gradient_bytes_each=parameters * 4,
        per_unit_summon_largest_block_extra_parameter_gradient_bytes_each=block * 4,
        hosts=hosts,
        GPU_admitted=False,
        CPU_oracle_admitted=False,
        excludes="activations, allocator retention, CPU audit scratch, runtime, communications, gather overlap",
        next="Qualify per-FSDP-unit ALL-gradient/AdamW audit on the existing352.96M references first. Plan streamed CPU gradients and explicit storage before admitting1.024B; do not run the unchanged full CPU oracle or deliberately OOM a single board.",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
