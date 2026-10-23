#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check ordinary packaged runtime selection, GPU autograd and frozen consumers."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--daily", action="store_true")
    parser.add_argument("--consumers", action="store_true")
    args = parser.parse_args()
    import torch
    import torch.distributed as dist

    result = dict(status="running", checks=[])
    try:
        torch.set_num_threads(1)
        assert torch.cuda.device_count() == 1
        assert torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1013:")
        assert dist.is_gloo_available()
        assert dist.is_nccl_available() == (not args.daily)
        result["checks"].append("native-device-and-explicit-backend-selection")
        if not args.daily:
            from bc250_torch import RCCL_SINGLE_ENVIRONMENT

            for name, value in RCCL_SINGLE_ENVIRONMENT.items():
                assert os.environ[name] == value, name
            assert os.environ["LD_LIBRARY_PATH"].split(":")[0] == "/workspace/rccl/lib"
            assert Path("/workspace/rccl/lib/librccl.so.1").is_file()
            # The runtime mount, not a test-only original path, supplies the DSO.
            assert any(
                "/workspace/rccl" in line and " ro," in line
                for line in Path("/proc/self/mountinfo").read_text().splitlines()
            )
            result["checks"].append("qualified-settings-and-readonly-packaged-library")
        for dtype in (torch.float32, torch.float64, torch.bfloat16):
            reference = (torch.arange(257) % 17 - 8).to(dtype) / 8
            value = reference.cuda().requires_grad_()
            weights = ((torch.arange(257) % 5) - 2).to(dtype) / 4
            output = value.square() * weights.cuda()
            output.sum().backward()
            torch.testing.assert_close(
                output.cpu(), reference.square() * weights, rtol=0, atol=0
            )
            torch.testing.assert_close(
                value.grad.cpu(), 2 * reference * weights, rtol=0, atol=0
            )
            result["checks"].append("exact-output-first-gradient-" + str(dtype))
        if args.consumers:
            import accelerate
            import peft
            import transformers

            assert (
                transformers.__version__,
                peft.__version__,
                accelerate.__version__,
            ) == ("5.2.0", "0.18.1", "1.12.0")
            result["checks"].append("frozen-consumers-import")
        torch.cuda.synchronize()
        result["status"] = "passed"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
