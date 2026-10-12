# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise the installed runtime repeatedly with CPU-checked results."""

import argparse
import json
from pathlib import Path
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("duration must be positive")

    import torch

    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1013"
    torch.set_num_threads(4)
    torch.manual_seed(1704)
    cases = []
    for dtype, tolerance in (
        (torch.float32, 5e-4),
        (torch.float16, 3e-3),
        (torch.bfloat16, 2e-2),
    ):
        a = (torch.randn(1024, 1024) / 32).to(dtype)
        b = (torch.randn(1024, 1024) / 32).to(dtype)
        reference = (a.double() @ b.double()).to(dtype)
        cases.append((a.cuda(), b.cuda(), reference, tolerance))
    host_values = torch.linspace(-1, 1, 65539)
    reference = torch.sin(host_values) * 2 + 3
    compiled = torch.compile(lambda x: torch.sin(x) * 2 + 3, fullgraph=True)
    torch.testing.assert_close(
        compiled(host_values.cuda()).cpu(), reference, rtol=1e-5, atol=1e-5
    )
    started = time.monotonic()
    cycles = 0

    def record(status: str) -> None:
        args.output.write_text(
            json.dumps(
                {
                    "status": status,
                    "torch": torch.__version__,
                    "arch": torch.cuda.get_device_properties(0).gcnArchName,
                    "seconds": time.monotonic() - started,
                    "cycles": cycles,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "reserved_bytes": torch.cuda.memory_reserved(),
                },
                indent=2,
            )
            + "\n"
        )

    record("running")
    while time.monotonic() - started < args.duration:
        for a, b, expected, tolerance in cases:
            torch.testing.assert_close(
                (a @ b).cpu(), expected, rtol=tolerance, atol=tolerance
            )
        storage = torch.full((4194307,), -17.0, device="cuda")
        view = storage[1:65540]
        view.copy_(host_values)
        view.copy_(compiled(view))
        torch.testing.assert_close(view.cpu(), reference, rtol=1e-5, atol=1e-5)
        assert storage[0].item() == -17 and storage[-1].item() == -17
        del view, storage
        cycles += 1
        if cycles % 20 == 0:
            torch.cuda.empty_cache()
            record("running")
            print(f"STABILITY_PROGRESS {cycles}", flush=True)
    torch.cuda.synchronize()
    record("passed")
    print(f"INSTALLED_RUNTIME_STABILITY_PASS cycles={cycles}", flush=True)


if __name__ == "__main__":
    main()
