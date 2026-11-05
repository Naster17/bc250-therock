#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Versioned, dtype/shape-sensitive C-order array digest for owned evidence fixtures."""
import hashlib
import json

import numpy as np


def array_digest(arrays, *, buffer_elements=65536):
    """Hash named NumPy arrays without Python lists or a whole contiguous copy.

    Hashes exact numeric bytes, including signed zeros/NaN payloads/endian dtype;
    storage layout does not affect logical C-order values. Not a legacy JSON
    fingerprint replacement with the same digest format. Object arrays refused.
    At most buffer_elements*itemsize is buffered for a non-contiguous input.
    """
    if type(buffer_elements) is not int or buffer_elements <= 0:
        raise ValueError("buffer_elements must be a positive integer")
    digest = hashlib.sha256(b"bc250-array-digest-v1\0")
    for name, value in sorted(arrays.items()):
        if not isinstance(name, str) or not isinstance(value, np.ndarray):
            raise ValueError("Expected string names and NumPy arrays")
        if value.dtype.hasobject or value.dtype.fields is not None:
            raise ValueError("Object and structured arrays are not evidence inputs")
        header = json.dumps(
            dict(name=name, shape=list(value.shape), dtype=value.dtype.str),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        iterator = np.nditer(
            value,
            flags=["external_loop", "buffered", "zerosize_ok"],
            op_flags=["readonly", "contig"],
            order="C",
            buffersize=buffer_elements,
        )
        for chunk in iterator:
            digest.update(memoryview(chunk).cast("B"))
    return digest.hexdigest()
