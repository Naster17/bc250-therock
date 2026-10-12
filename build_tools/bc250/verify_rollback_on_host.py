#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify both versioned releases through the launcher and restore the new one."""

import json
import os
from pathlib import Path
import subprocess

from deploy_on_host import switch_link


def main() -> None:
    if os.geteuid() != 0:
        raise RuntimeError("Run rollback verification as root")
    base = Path("/opt/bc250-stack")
    current = base / "current"
    candidate = current.resolve(strict=True)
    previous = (base / "previous").resolve(strict=True)
    if candidate == previous:
        raise RuntimeError("Expected distinct current and previous releases")
    for release in (candidate, previous):
        if not json.loads((release / "runtime.json").read_text())["validated"]:
            raise RuntimeError(f"Release is not validated: {release}")
    probe = (
        "import torch; "
        "assert torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]=='gfx1013'; "
        "a=torch.arange(4096,dtype=torch.float32).reshape(64,64)/4096; "
        "torch.testing.assert_close((a.cuda()@a.cuda()).cpu(),a@a,rtol=1e-4,atol=1e-4); "
        "print('ROLLBACK_NATIVE_GEMM_PASS')"
    )

    def check() -> None:
        subprocess.run(
            ["/usr/local/bin/bc250-torch", "-c", probe],
            stdin=subprocess.DEVNULL,
            check=True,
            timeout=120,
        )

    try:
        switch_link(current, previous)
        check()
    finally:
        switch_link(current, candidate)
    check()
    (candidate / "validation/rollback.json").write_text(
        json.dumps(
            {"status": "passed", "current": str(candidate), "previous": str(previous)},
            indent=2,
        )
        + "\n"
    )
    print(f"ROLLBACK_VERIFIED_AND_CURRENT_RESTORED {candidate}", flush=True)


if __name__ == "__main__":
    main()
