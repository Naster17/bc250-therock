#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Load actual B05.5 checkpoints through public W8 API and reverse the roundtrip."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from bc250_quant import FrozenW8Linear
    from bc250_quant_b055 import FrozenW8Linear as Legacy

    torch.set_num_threads(1)
    report = dict(status="running", checks=[])
    try:
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for policy in ("none", "frozen", "trainable"):
                file = (
                    Path(__file__).resolve().parent.parent
                    / "logs/quant-b055-state-r1"
                    / f"state-{str(dtype)}-{policy}.pt"
                )
                state = torch.load(file, map_location="cpu", weights_only=True)
                old = Legacy.from_state_dict(state, device="cuda")
                new = FrozenW8Linear.from_state_dict(state, device="cuda")
                x = (
                    (torch.arange(34, device="cuda").reshape(2, 17) / 64)
                    .to(dtype)
                    .requires_grad_()
                )
                y = x.detach().clone().requires_grad_()
                a, b = old(x), new(y)
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                a.sum().backward()
                b.sum().backward()
                torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
                if policy == "trainable":
                    torch.testing.assert_close(
                        old.bias.grad, new.bias.grad, rtol=0, atol=0
                    )
                saved = args.output.parent / file.name
                torch.save(new.cpu().state_dict(), saved)
                reverse = Legacy.from_state_dict(
                    torch.load(saved, map_location="cpu", weights_only=True),
                    device="cuda",
                )
                torch.testing.assert_close(reverse(x.detach()), a, rtol=0, atol=0)
                report["checks"].append(
                    "b055-file-public-output-first-grads-reverse-"
                    + str(dtype)
                    + "-"
                    + policy
                )
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
