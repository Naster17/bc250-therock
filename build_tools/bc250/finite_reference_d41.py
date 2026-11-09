#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded little-endian IEEE float finiteness and SHA audit, without Torch/model RAM."""
import hashlib
import os
from pathlib import Path
import stat
from artifact_d35 import safe_path


def check_chunk(data, width):
    if width not in {4, 8} or len(data) % width:
        raise ValueError("Invalid floating-point blob alignment/width")
    # The high seven exponent bits share a byte with the sign. Check only
    # candidates with all seven set, then the remaining exponent bits below it.
    high = data[width - 1 :: width]
    mask = 0x80 if width == 4 else 0xF0
    for marker in [b"\x7f", b"\xff"]:
        start = 0
        while True:
            index = high.find(marker, start)
            if index < 0:
                break
            if data[width * index + width - 2] & mask == mask:
                raise ValueError("Nonfinite CPU reference value")
            start = index + 1


def scan(root, item, width):
    path = safe_path(Path(root), item["file"])
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size != item["bytes"]:
            raise ValueError("Reference file type/length changed")
        digest = hashlib.sha256()
        count = 0
        while True:
            data = os.read(fd, 1024**2)
            if not data:
                break
            check_chunk(data, width)
            digest.update(data)
            count += len(data)
        if count != item["bytes"] or digest.hexdigest() != item["sha256"]:
            raise ValueError("Reference bytes/SHA changed")
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return count // width
    finally:
        os.close(fd)
