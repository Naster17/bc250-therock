#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Retained CPU contracts and same-device optimized metric validation."""
import torch
from bc250_compact_token_metrics import CompactTokenMetrics
from d18_compact_reference import CompactTokenMetrics as D18Compact
from check_token_metrics_d18 import contract_checks as d18_contract_checks


def GPU_contract_checks(device):
    z = (
        torch.arange(2 * 5 * 7, device=device, dtype=torch.float32).reshape(2, 5, 7)
        / 31
    )
    y = torch.arange(10, device=device).reshape(2, 5) % 7
    y[0, :3] = -100
    z[0, 0, 0] = float("nan")
    z[0, 1, 1] = float("inf")
    for causal in [False, True]:
        callback = CompactTokenMetrics(causal=causal, chunk_tokens=3)
        torch.testing.assert_close(
            callback.preprocess(z, y),
            D18Compact(causal=causal, chunk_tokens=3).preprocess(z, y),
            atol=1e-12,
            rtol=1e-12,
        )
        for kind in ["nan", "inf", "negative", "large"]:
            values = z.clone()
            labels = y.clone()
            if kind in ["nan", "inf"]:
                values[1, 1, 0] = float(kind)
            else:
                labels[1, 2] = -2 if kind == "negative" else 7
            try:
                callback.preprocess(values, labels)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid active input accepted")
    return [
        "masked-and-causal-ignored-nonfinites-parity",
        "eight-active-nonfinite-and-label-range-refusals",
    ]


def contract_checks():
    checks = d18_contract_checks()
    GPU_contract_checks(torch.device("cpu"))
    checks.append("ignored-nonfinite-parity-and-eight-active-invalid-refusals")
    return checks
