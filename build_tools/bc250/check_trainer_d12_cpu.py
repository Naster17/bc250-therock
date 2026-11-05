#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CPU-only loss derivative and pinned-source/version admission checks."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import accelerate
import transformers
from bc250_trainer import BC250Trainer, masked_token_loss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    assert not torch.cuda.is_initialized()
    checks = []
    before = torch.get_rng_state().clone()
    for module in [transformers, accelerate]:
        with patch.object(module, "__version__", "unqualified"):
            try:
                BC250Trainer.create_accelerator_and_postprocess(object())
            except RuntimeError as e:
                assert "requalification" in str(e)
            else:
                raise AssertionError("Future package accepted")
        checks.append("version-refusal-" + module.__name__)
    read = Path.read_bytes
    for name in ["trainer.py", "accelerator.py"]:

        def tampered(path):
            return b"same-version-different-source" if path.name == name else read(path)

        with patch.object(Path, "read_bytes", tampered):
            try:
                BC250Trainer.create_accelerator_and_postprocess(object())
            except RuntimeError as e:
                assert "source identity" in str(e)
            else:
                raise AssertionError("Changed same-version source accepted")
        checks.append("source-refusal-" + name)
    assert torch.equal(before, torch.get_rng_state())
    values = torch.tensor(
        [[[0.7, -0.2, 0.4], [0.1, 0.2, -0.8]], [[-0.4, 0.3, 0.9], [0.2, 0.6, -0.1]]],
        dtype=torch.float64,
    )
    labels = torch.tensor([[2, -100], [0, 1]])
    for denominator in [None, 23]:
        x = values.clone().requires_grad_()
        loss = masked_token_loss(SimpleNamespace(logits=x), labels, denominator)
        active = labels.ne(-100)
        flat = x.detach()[active]
        target = labels[active]
        divisor = 3 if denominator is None else denominator
        expected = (
            flat.logsumexp(-1) - flat.gather(1, target[:, None]).squeeze(1)
        ).sum() / divisor
        loss.backward()
        derivative = torch.zeros_like(x)
        z = flat.softmax(-1)
        z[torch.arange(3), target] -= 1
        derivative[active] = z / divisor
        torch.testing.assert_close(loss.detach(), expected, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(x.grad, derivative, atol=1e-14, rtol=1e-14)
        checks.append("CPU-masked-CE-loss-and-analytic-gradient-" + str(denominator))
    for count in [0, -1]:
        try:
            masked_token_loss(SimpleNamespace(logits=values), labels, count)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid global token count accepted")
        checks.append("invalid-token-count-" + str(count))
    assert not torch.cuda.is_initialized()
    checks.append("CPU-only-Torch-CUDA-not-initialized")
    a.output.write_text(
        json.dumps(
            dict(
                status="passed",
                checks=checks,
                torch_cuda_initialized=False,
                helper_sha256=hashlib.sha256(
                    Path(__file__).with_name("bc250_trainer.py").read_bytes()
                ).hexdigest(),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
