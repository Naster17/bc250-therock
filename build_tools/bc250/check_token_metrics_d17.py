#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CPU semantic checks of gathered masked/causal token metrics."""
from types import SimpleNamespace
import math

import numpy as np
from bc250_token_metrics import TokenWeightedMetrics


def manual_metrics(logits, labels, causal=False):
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels)
    if causal:
        z = z[:, :-1]
        y = y[:, 1:]
    active = y != -100
    z = z[active]
    y = y[active]
    maximum = z.max(-1)
    summed = (
        maximum + np.log(np.exp(z - maximum[:, None]).sum(-1)) - z[np.arange(len(y)), y]
    ).sum()
    return dict(
        token_nll=float(summed / len(y)),
        token_perplexity=math.exp(float(summed / len(y))),
        token_accuracy=float((z.argmax(-1) == y).sum() / len(y)),
        active_tokens=len(y),
        evaluated_examples=len(labels),
    )


def contract_checks():
    z = np.array(
        [
            [[0.7, -0.2, 0.4], [0.1, 0.2, -0.8], [-0.4, 0.3, 0.9]],
            [[0.2, 0.6, -0.1], [-0.5, 0.4, 0.2], [0.6, -0.3, 0.8]],
        ],
        dtype=np.float32,
    )
    y = np.array([[2, -100, 1], [-100, 2, 0]], dtype=np.int64)
    checks = []
    for causal in [False, True]:
        prediction = SimpleNamespace(predictions=z, label_ids=y)
        expected = manual_metrics(z, y, causal)
        for chunk in [1, 4096]:
            actual = TokenWeightedMetrics(causal=causal, chunk_tokens=chunk)(prediction)
            for key, value in expected.items():
                assert math.isclose(actual[key], value, rel_tol=1e-12, abs_tol=1e-12)
        checks.append(
            ("causal" if causal else "masked")
            + "-manual-CE-count-accuracy-and-chunk-equivalence"
        )
    prediction = SimpleNamespace(predictions=z, label_ids=y)
    callback = TokenWeightedMetrics()
    assert callback(prediction) == callback(prediction)
    checks.append("stateless-repeat-no-metric-leak")
    for labels, logits, label in [
        (np.full_like(y, -100), z, "empty-target-refusal"),
        (y, z.argmax(-1), "argmax-only-input-refusal"),
    ]:
        try:
            callback(SimpleNamespace(predictions=logits, label_ids=labels))
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid metric input accepted")
        checks.append(label)
    return checks
