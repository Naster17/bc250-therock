#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Explicit D41 model dimensions with unchanged qualified D35 construction/math."""
import fixture_d35 as original
from capacity_plan_d41 import CONFIG, CONTROL


def select(shape):
    if shape not in {"control", "large"}:
        raise ValueError("Unknown fixture shape")
    original.CONFIG = dict(CONTROL if shape == "control" else CONFIG)
    return original
