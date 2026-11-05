#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Limits accepted/refused before FSDP clipping setup; no GPU scalar coercion."""
import torch
from bc250_trainer_fsdp import _clip_limit


def contracts():
    for value in [0, 0.05, 1.0, 100]:
        assert _clip_limit(value) == float(value)
    for value in [
        -1,
        float("nan"),
        float("inf"),
        None,
        True,
        "0.05",
        torch.tensor(0.05),
    ]:
        try:
            _clip_limit(value)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid clipping limit accepted")
    return [
        "finite-nonnegative-limits",
        "negative-nonfinite-null-bool-string-tensor-refusals",
    ]
