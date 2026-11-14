#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PLAN2-01 benign kernel-noise filter for the soak guardian (PRE-DECLARED).

The host's cyan-skillfish governor queries GPU clock types that gfx1013 does not
support, logging benign `amdgpu ...: Unsupported clock type` warnings (observed
2026-11-14: a 4-line burst during soak seg2, training otherwise perfect). These
are clock-query noise, NOT GPU faults. The soak guardian's kernel-log identity
check filters ONLY this exact substring; ANY other dmesg change (real GPU fault,
ring timeout, reset, page/VM fault, panic, foreign activity) still fails the run.

This is a deliberate, pre-declared refinement of the D69/D70/D71 kernel-log
invariant for this host's governor behavior. The original qualified guardians
are unchanged; only the soak copy uses this filter. Real-fault detection is
preserved because no genuine GPU fault message contains "Unsupported clock type".
"""

from __future__ import annotations

BENIGN_KERNEL_PATTERNS = ("Unsupported clock type",)


def _benign(line: bytes) -> bool:
    return any(pattern.encode() in line for pattern in BENIGN_KERNEL_PATTERNS)


def filter_benign(data: bytes) -> list[bytes]:
    return [line for line in data.splitlines() if not _benign(line)]


def kernel_log_equivalent(before: bytes, after: bytes) -> bool:
    """True iff the only kernel-log difference is benign clock-query noise."""
    return filter_benign(before) == filter_benign(after)


def benign_count(data: bytes) -> int:
    return sum(1 for line in data.splitlines() if _benign(line))
