#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CPU compact statistics contracts and independent NumPy reference."""
from types import SimpleNamespace
import math

import numpy as np
import torch
from bc250_compact_token_metrics import CompactTokenMetrics
from bc250_token_metrics import TokenWeightedMetrics
from check_token_metrics_d17 import manual_metrics


def contract_checks():
    g = torch.Generator().manual_seed(1818)
    z = torch.randn(5, 7, 13, generator=g)
    y = torch.randint(13, (5, 7), generator=g)
    y[0] = -100
    y[2, :3] = -100
    checks = []
    for causal in [False, True]:
        expected = manual_metrics(z.numpy(), y.numpy(), causal)
        for chunk in [1, 3, 4096]:
            callback = CompactTokenMetrics(causal=causal, chunk_tokens=chunk)
            stats = callback.preprocess(z, y)
            assert stats.shape == (5, 3) and torch.equal(
                stats[0], torch.zeros(3).double()
            )
            prediction = SimpleNamespace(predictions=stats, label_ids=y)
            actual = callback(prediction)
            assert actual == callback(prediction)
            full = TokenWeightedMetrics(causal=causal)(
                SimpleNamespace(predictions=z, label_ids=y)
            )
            for key, value in expected.items():
                assert math.isclose(actual[key], value, rel_tol=1e-12, abs_tol=1e-12)
                assert math.isclose(
                    actual[key], full[key], rel_tol=1e-12, abs_tol=1e-12
                )
        checks.append(
            ("causal" if causal else "masked")
            + "-chunk-stateless-zero-row-manual-full-parity"
        )
    # Emulate native duplicated tail: per-example rows must be sliced before sum.
    callback = CompactTokenMetrics()
    stats = callback.preprocess(z, y)
    assert callback(
        SimpleNamespace(predictions=torch.cat([stats, stats[:1]])[:5], label_ids=y)
    ) == callback(SimpleNamespace(predictions=stats, label_ids=y))
    checks.append("per-example-tail-slice-before-total")
    bad = stats.clone()
    bad[1, 1] += 1
    bad_correct = stats.clone()
    bad_correct[1, 2] = 0.5
    bad_nan = z.clone()
    bad_nan[1, 1, 0] = float("nan")
    bad_y = y.clone()
    bad_y[1, 1] = 13
    attempts = [
        lambda: callback.preprocess(z.argmax(-1), y),
        lambda: callback.preprocess(z, None),
        lambda: callback.preprocess(bad_nan, y),
        lambda: callback.preprocess(z, bad_y),
        lambda: callback(SimpleNamespace(predictions=bad, label_ids=y)),
        lambda: callback(SimpleNamespace(predictions=bad_correct, label_ids=y)),
        lambda: callback(
            SimpleNamespace(
                predictions=callback.preprocess(z, torch.full_like(y, -100)),
                label_ids=torch.full_like(y, -100),
            )
        ),
    ]
    for attempt in attempts:
        try:
            attempt()
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid compact input accepted")
    checks.append("seven-invalid-shape-label-logit-count-empty-refusals")
    assert np.isfinite(stats.numpy()).all()
    return checks
