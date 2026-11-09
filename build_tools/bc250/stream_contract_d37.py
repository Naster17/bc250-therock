#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fail-closed completion and one-tensor disk admission for a CPU oracle."""
GIB = 1024**3


class Completion:
    def __init__(self, names):
        self.names = list(names)
        if not self.names or len(self.names) != len(set(self.names)):
            raise ValueError("Missing/duplicate reference names")
        self.seen = set()

    def accept(self, name):
        if name not in self.names or name in self.seen:
            raise ValueError("Unexpected/repeated gradient completion")
        self.seen.add(name)

    def finish(self):
        if self.seen != set(self.names):
            raise ValueError("Incomplete gradient coverage")


def admit_scratch(free_bytes, max_tensor_bytes):
    if type(max_tensor_bytes) is not int or max_tensor_bytes <= 0:
        raise ValueError("Invalid scratch bound")
    if free_bytes < 5 * GIB + max_tensor_bytes:
        raise ValueError("Insufficient one-tensor scratch plus5GiB reserve")
    return dict(
        free_before_bytes=free_bytes,
        maximum_scratch_bytes=max_tensor_bytes,
        minimum_free_after_bytes=5 * GIB,
    )
