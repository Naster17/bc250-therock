#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Ordinary-user caller context, script semantics and native backward fixture."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sentinel", required=True)
    args = p.parse_args()
    r = dict(status="running", checks=[])
    try:
        assert os.geteuid() == 1000
        assert os.environ["BC250_CK_BARE"] == "caller-value"
        assert os.environ["BC250_CK_FILE"] == "file-value"
        assert args.sentinel == "argument with spaces"
        assert sys.path[0] == str(Path(__file__).resolve().parent)
        assert (
            os.environ["MIOPEN_CK_LIB_PATH"]
            == "/opt/bc250-stack/sdks/bc250-miopen-dl-gfx1013-b0712/lib"
        )
        r["checks"].append(
            "ordinary-UID1000-bare-env-envfile-argv-script-path-and-installed-activation"
        )
        import torch

        assert torch.cuda.get_device_properties(0).gcnArchName == "gfx1013:xnack-"
        x = torch.arange(35, device="cuda", dtype=torch.float32).requires_grad_()
        x.square().sum().backward()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            x.grad.cpu(), 2 * torch.arange(35, dtype=torch.float32), rtol=0, atol=0
        )
        r["checks"].append("ordinary-user-exact-native-backward")
        assert len(r["checks"]) == 2
        r["status"] = "passed"
    except BaseException as error:
        r.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(r, indent=2) + "\n")


if __name__ == "__main__":
    main()
