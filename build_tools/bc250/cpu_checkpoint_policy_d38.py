#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exact native non-reentrant checkpoint recipe; no truthy integer substitutions."""
from functools import partial

OPTIONS = dict(use_reentrant=False, preserve_rng_state=True)


def validate_options(options):
    if (
        type(options) is not dict
        or options != OPTIONS
        or any(type(v) is not bool for v in options.values())
    ):
        raise ValueError("Explicit non-reentrant RNG-preserving CPU recipe required")


def callable_matches(function, native):
    if (
        not isinstance(function, partial)
        or function.func is not native
        or function.args
    ):
        return False
    try:
        validate_options(function.keywords)
    except ValueError:
        return False
    return True
