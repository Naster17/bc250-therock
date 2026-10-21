#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Read ELF text without objcopy's default in-place output behavior."""

import hashlib
import json
import struct
from qualify_training_n33_on_host import WS


def section(path, name):
    data = path.read_bytes()
    assert data[:6] == b"\x7fELF\x02\x01"
    offset = struct.unpack_from("<Q", data, 40)[0]
    entry_size, count, names = struct.unpack_from("<HHH", data, 58)
    headers = [
        struct.unpack_from("<IIQQQQIIQQ", data, offset + i * entry_size)
        for i in range(count)
    ]
    strings = headers[names]
    strings = data[strings[4] : strings[4] + strings[5]]
    for header in headers:
        label = strings[header[0] :].split(b"\0")[0].decode()
        if label == name:
            return data[header[4] : header[4] + header[5]], header[3]
    raise ValueError(name)


root = WS / "audio-b042/hip-symbols"
report = {}
for label, path in (
    ("qualified", WS / "build/dist/rocm/lib/libamdhip64.so"),
    ("symbols", root / "libamdhip64-symbols.so"),
):
    text, address = section(path, ".text")
    report[label] = dict(
        sha256=hashlib.sha256(text).hexdigest(), address=address, size=len(text)
    )
assert report["qualified"] == report["symbols"], report
(root / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
print("HIP_TEXT_IDENTICAL", report)
