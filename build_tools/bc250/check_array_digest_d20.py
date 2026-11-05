#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Independent whole-byte oracle and representation-sensitive digest contracts."""
import hashlib
import json

import numpy as np
from bc250_array_digest import array_digest


def contracts():
    value = np.arange(5 * 7, dtype=np.float32).reshape(5, 7)[:, ::-2]
    arrays = dict(logits=value, labels=np.arange(5, dtype=np.int64))
    oracle = hashlib.sha256(b"bc250-array-digest-v1\0")
    for name, x in sorted(arrays.items()):
        header = json.dumps(
            dict(name=name, shape=list(x.shape), dtype=x.dtype.str),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        oracle.update(len(header).to_bytes(8, "little"))
        oracle.update(header)
        oracle.update(x.tobytes(order="C"))
    for cap in [1, 3, 65536]:
        assert array_digest(arrays, buffer_elements=cap) == oracle.hexdigest()
        assert (
            array_digest(
                dict(labels=arrays["labels"], logits=value.copy()), buffer_elements=cap
            )
            == oracle.hexdigest()
        )
    assert array_digest({"x": np.array([], dtype=np.float32)}) != array_digest(
        {"x": np.array([], dtype=np.float64)}
    )
    assert array_digest({"x": np.array([0.0], dtype=np.float32)}) != array_digest(
        {"x": np.array([-0.0], dtype=np.float32)}
    )
    assert array_digest({"x": np.arange(6, dtype=np.int64)}) != array_digest(
        {"x": np.arange(6, dtype=np.int64).reshape(2, 3)}
    )
    for arrays in [
        {"x": np.array([object()])},
        {"x": np.zeros(2, dtype=[("a", "i4")])},
    ]:
        try:
            array_digest(arrays)
        except ValueError:
            pass
        else:
            raise AssertionError("Unsupported array accepted")
    return [
        "whole-byte-oracle-strided-and-chunk-order-parity",
        "dtype-shape-and-signed-zero-sensitive",
        "object-and-structured-refusals",
    ]
