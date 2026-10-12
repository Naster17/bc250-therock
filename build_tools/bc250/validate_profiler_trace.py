# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check rocprofv3 JSON from the CPU-verified native HIP smoke workload."""

import argparse
from collections import Counter
import json
from pathlib import Path


def validate(path: Path) -> None:
    data = json.loads(path.read_text())["rocprofiler-sdk-tool"]
    if isinstance(data, list):
        assert len(data) == 1, "expected one profiled process"
        data = data[0]
    records = data["buffer_records"]
    apis = records["hip_api"]
    kernels = records["kernel_dispatch"]
    copies = records["memory_copy"]
    assert apis and kernels and copies, "missing requested tracing domain"
    for domain in (apis, kernels, copies):
        for record in domain:
            assert 0 < record["start_timestamp"] <= record["end_timestamp"], record
            assert record["correlation_id"]["internal"] > 0, record
    symbols = data["kernel_symbols"]
    counts: Counter[str] = Counter()
    api_ids = {record["correlation_id"]["internal"] for record in apis}
    for record in kernels:
        dispatch = record["dispatch_info"]
        symbol = symbols[dispatch["kernel_id"]]
        name = symbol["formatted_kernel_name"]
        assert dispatch["queue_id"]["handle"] > 0, dispatch
        assert dispatch["agent_id"]["handle"] > 0, dispatch
        if name.startswith("vector_add("):
            assert record["correlation_id"]["internal"] in api_ids, record
            assert dispatch["workgroup_size"]["x"] == 256, dispatch
            counts["vector_add"] += 1
        elif name.startswith("lds_and_atomic("):
            assert record["correlation_id"]["internal"] in api_ids, record
            counts["lds_and_atomic"] += 1
    assert counts == {"vector_add": 256, "lds_and_atomic": 1}, counts
    print(
        "ROCPROFILER_NATIVE_TRACE_PASS",
        json.dumps(
            {"hip_apis": len(apis), "dispatches": len(kernels), "copies": len(copies)}
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="rocprofv3 JSON output")
    args = parser.parse_args()
    validate(args.trace)


if __name__ == "__main__":
    main()
