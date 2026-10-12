# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bounded UMA pressure, streams/events, allocator reuse and process gates."""

import argparse
import gc
import json
from pathlib import Path
import time

import torch

GIB = 1024**3
CHUNK = 256 * 1024**2


def available_memory() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Cannot determine the host's shared-memory budget")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("pressure", "streams", "worker"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-gib", type=int, default=10)
    parser.add_argument("--worker", type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.max_gib <= 10:
        parser.error("The SSH-safe qualification caps GPU pressure at 10 GiB")
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1013"
    torch.set_num_threads(4)
    report = {
        "status": "running",
        "mode": args.mode,
        "worker": args.worker,
        "checks": [],
    }
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()

    def passed(name: str) -> None:
        torch.cuda.synchronize()
        report["checks"].append(name)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"MEMORY_GATE_PASS {args.mode} {name}", flush=True)

    if args.mode == "pressure":
        if available_memory() < 6 * GIB:
            raise RuntimeError(
                "Insufficient free shared RAM for safe pressure qualification"
            )
        host = torch.empty(768 * 1024**2, dtype=torch.uint8)
        host.fill_(173)
        # CPU and GPU use the same physical memory; retain 3 GiB for SSH/kernel/scratch.
        free, total = torch.cuda.mem_get_info()
        safe = min(available_memory(), free) - 3 * GIB
        chunks = min(args.max_gib * GIB, safe) // CHUNK
        if chunks < 8:
            raise RuntimeError(
                "Shared-memory headroom fell below the safe test threshold"
            )
        storage = []
        for index in range(chunks):
            tensor = torch.empty(CHUNK, device="cuda", dtype=torch.uint8)
            tensor.fill_((index * 17 + 3) % 251)
            storage.append(tensor)
        torch.cuda.synchronize()
        for index, tensor in enumerate(storage):
            assert torch.all(tensor == (index * 17 + 3) % 251).item()
        assert torch.all(host == 173).item()
        report.update(
            gpu_pressure_bytes=chunks * CHUNK,
            cpu_pressure_bytes=host.numel(),
            gpu_advertised_total=total,
            host_available_during_pressure=available_memory(),
        )
        passed("all_bytes_verified_under_bounded_shared_cpu_gpu_pressure")
        # Keep alternate allocations live while freeing/reusing holes.
        for index in range(0, len(storage), 2):
            storage[index] = None
        gc.collect()
        torch.cuda.empty_cache()
        for index in range(0, len(storage), 2):
            tensor = torch.empty(CHUNK, device="cuda", dtype=torch.uint8)
            tensor.fill_(229)
            assert torch.all(tensor == 229).item()
            del tensor
        for index in range(1, len(storage), 2):
            assert torch.all(storage[index] == (index * 17 + 3) % 251).item()
        passed("fragmented_allocator_reuse_preserves_live_allocations")
        del storage, tensor, host
    elif args.mode == "streams":
        values = torch.linspace(-1, 1, 1048583)
        expected = values * 3 + 7
        streams = [torch.cuda.Stream() for _ in range(4)]
        for iteration in range(12):
            outputs, events = [], []
            for stream in streams:
                with torch.cuda.stream(stream):
                    source = values.pin_memory().to("cuda", non_blocking=True)
                    output = source * 3 + 7
                    event = torch.cuda.Event()
                    event.record(stream)
                    outputs.append(output)
                    events.append(event)
            for event, output in zip(events, outputs):
                torch.cuda.current_stream().wait_event(event)
                torch.testing.assert_close(output.cpu(), expected, rtol=1e-6, atol=1e-6)
            del source, output, outputs, events
        passed("four_streams_pinned_async_copies_events_and_cpu_results")
        graph_stream = torch.cuda.Stream()
        static_input = values.cuda()
        graph_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(graph_stream):
            for _ in range(3):
                static_output = static_input * 3 + 7
        torch.cuda.current_stream().wait_stream(graph_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=graph_stream):
            static_output = static_input * 3 + 7
        for iteration in range(8):
            static_input.copy_(values + iteration)
            graph.replay()
            torch.testing.assert_close(
                static_output.cpu(), (values + iteration) * 3 + 7, rtol=1e-6, atol=1e-6
            )
        passed("hip_graph_capture_and_eight_cpu_checked_replays")
        del graph, static_input, static_output
    else:
        # Two independent instances run together, with distinguishable byte patterns.
        marker = 31 + args.worker
        guard = torch.full((256 * 1024**2,), marker, dtype=torch.uint8, device="cuda")
        torch.manual_seed(1704 + args.worker)
        a, b = torch.randn(129, 65), torch.randn(65, 97)
        expected = (a.double() @ b.double()).float()
        gpu_a, gpu_b = a.cuda(), b.cuda()
        for iteration in range(12):
            torch.testing.assert_close(
                (gpu_a @ gpu_b).cpu(), expected, rtol=5e-4, atol=1e-4
            )
            scratch = torch.full(
                (64 * 1024**2,),
                (marker + iteration) % 251,
                dtype=torch.uint8,
                device="cuda",
            )
            assert torch.all(scratch == (marker + iteration) % 251).item()
            assert torch.all(guard == marker).item()
            del scratch
        passed("independent_process_gemm_allocations_and_live_buffer_guards")
        del guard, gpu_a, gpu_b
    gc.collect()
    torch.cuda.synchronize()
    report.update(
        status="passed",
        seconds=time.monotonic() - started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        allocated_after_bytes=torch.cuda.memory_allocated(),
        baseline_allocated_bytes=baseline,
    )
    # A few loop locals may retain small tensors; reject unexpected large leaks.
    assert report["allocated_after_bytes"] - baseline < 64 * 1024**2
    torch.cuda.empty_cache()
    report["reserved_after_empty_cache_bytes"] = torch.cuda.memory_reserved()
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
