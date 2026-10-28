#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check ordinary opt-in activation, external script paths and runtime coexistence."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile", choices=("daily", "rccl-single"), required=True)
    args = parser.parse_args()
    import migraphx
    import torch
    import numpy as np

    sdk = Path(os.environ["BC250_MIGRAPHX_SDK"])
    checks = []
    assert Path(migraphx.__file__).parent == sdk / "lib"
    checks.append("installed-binding-path")
    assert sys.version_info[:2] == (3, 12)
    checks.append("qualified-cpython")
    mount = next(
        line
        for line in Path("/proc/self/mountinfo").read_text().splitlines()
        if line.split()[4] == str(sdk)
    )
    assert "ro" in mount.split()[5].split(",")
    checks.append("sdk-read-only-mount")
    assert (
        Path(__file__).with_name("sidecar.json").read_text()
        == '{"owned_fixture": true}\n'
    )
    assert "application with spaces" in str(Path(__file__).parent)
    checks.append("external-script-parent-and-space-path")
    assert os.environ["HSA_ENABLE_SDMA"] == "0"
    assert os.environ.get("BC250_B068_MARKER") == "ordinary activation"
    checks.append("safe-default-and-explicit-environment")
    assert torch.distributed.is_nccl_available() == (args.profile == "rccl-single")
    checks.append("requested-torch-runtime-cohort")
    x = torch.arange(17, device="cuda", dtype=torch.float32).requires_grad_()
    x.square().sum().backward()
    np.testing.assert_array_equal(
        x.grad.cpu().numpy(), 2 * np.arange(17, dtype=np.float32)
    )
    torch.cuda.synchronize()
    checks.append("torch-native-first-backward-with-sdk")
    import onnxruntime as ort

    assert not {"MIGraphXExecutionProvider", "ROCMExecutionProvider"} & set(
        ort.get_available_providers()
    )
    checks.append("ort-gpu-provider-unregistered")
    args.output.write_text(
        json.dumps(
            dict(
                status="passed",
                checks=checks,
                sdk=str(sdk),
                profile=args.profile,
                mount=mount,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
