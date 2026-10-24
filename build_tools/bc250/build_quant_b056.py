#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a deterministic pure-Python wheel from the pinned qualified W8 sources."""

import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

LICENSE = 'MIT License\n\nCopyright (c) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\nof this software and associated documentation files (the "Software"), to deal\nin the Software without restriction, including without limitation the rights\nto use, copy, modify, merge, publish, distribute, sublicense, and/or sell\ncopies of the Software, and to permit persons to whom the Software is\nfurnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all\ncopies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\nIMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\nFITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\nAUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\nLIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\nOUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\nSOFTWARE.\n'

VERSION = "0.1.0"
DIST = "bc250_quant-" + VERSION + ".dist-info"
PINNED = {
    "bc250_quant_b051.py": "6832ff9e58104236d877a455fdfe90f11b3803198732497d6dd4ac77242fd319",
    "bc250_quant_b055.py": "a18e988fab0e9cf816c94cd21cffdb151661a443a77a14c92d5cdd02f0bbd168",
}


def build(source, output):
    payload = {}
    for name, digest in PINNED.items():
        data = (source / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, name
        if name.endswith("b051.py"):
            payload["bc250_quant/_w8.py"] = data
        else:
            old = b"import bc250_quant_b051 as quant"
            assert data.count(old) == 1
            payload["bc250_quant/modules.py"] = data.replace(
                old, b"from . import _w8 as quant"
            )
    payload["bc250_quant/__init__.py"] = (
        b'"""Opt-in bounded frozen W8; no automatic dispatch/model replacement."""\nfrom ._w8 import FrozenInt8Weight, pack, linear\nfrom .modules import FrozenW8Linear\n\n__version__ = "0.1.0"\n__all__ = ["FrozenInt8Weight", "FrozenW8Linear", "pack", "linear"]\n'
    )
    payload["bc250_quant/provenance.json"] = (
        json.dumps(
            dict(
                source_sha256=PINNED,
                module_state_format=1,
                qualified_scope="Bounded eager native gfx1013 frozen W8 first derivatives; explicit opt-in",
            ),
            indent=2,
        )
        + "\n"
    ).encode()
    payload[DIST + "/METADATA"] = (
        "Metadata-Version: 2.1\nName: bc250-quant\nVersion: "
        + VERSION
        + "\nSummary: Explicit bounded gfx1013 frozen W8 module\n"
        "Requires-Python: >=3.12\nRequires-Dist: torch\nLicense: MIT\n\n"
    ).encode()
    payload[DIST + "/licenses/LICENSE"] = LICENSE.encode()
    payload[DIST + "/WHEEL"] = (
        b"Wheel-Version: 1.0\nGenerator: bc250-b056\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    )
    records = io.StringIO(newline="")
    writer = csv.writer(records, lineterminator="\n")
    for name, data in sorted(payload.items()):
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(data).digest())
            .rstrip(b"=")
            .decode()
        )
        writer.writerow([name, "sha256=" + digest, len(data)])
    writer.writerow([DIST + "/RECORD", "", ""])
    payload[DIST + "/RECORD"] = records.getvalue().encode()
    output.mkdir(parents=True, exist_ok=True)
    wheel = output / ("bc250_quant-" + VERSION + "-py3-none-any.whl")
    assert not wheel.exists()
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 24, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    return wheel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.source, args.output))
