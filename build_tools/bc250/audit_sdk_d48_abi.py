#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare SDK dynamic symbol contracts/DT_NEEDED without loading either DSO."""
import argparse
import json
from pathlib import Path
import struct

from audit_elf_d46 import elf


def contract(data):
    info = elf(data)
    sections = info["sections"]
    strings = sections[".dynstr"]
    table = data[strings["offset"] : strings["offset"] + strings["size"]]

    def name(index):
        assert index < len(table)
        return table[index : table.index(b"\0", index)].decode()

    symbols = sections[".dynsym"]
    assert symbols["size"] % 24 == 0
    entries = []
    for offset in range(symbols["offset"], symbols["offset"] + symbols["size"], 24):
        index, flags, visibility, section, value, size = struct.unpack_from(
            "<IBBHQQ", data, offset
        )
        if flags >> 4:
            entries.append(
                (name(index), flags >> 4, flags & 15, visibility & 3, bool(section))
            )
    needed, soname = [], []
    dynamic = sections[".dynamic"]
    for offset in range(dynamic["offset"], dynamic["offset"] + dynamic["size"], 16):
        tag, value = struct.unpack_from("<qQ", data, offset)
        if tag == 1:
            needed.append(name(value))
        if tag == 14:
            soname.append(name(value))
    return dict(symbols=sorted(entries), needed=needed, soname=soname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    args = parser.parse_args()
    old, new = contract(args.old.read_bytes()), contract(args.new.read_bytes())
    assert old == new, "SDK dynamic symbol/dependency contract changed"
    print(
        json.dumps(
            dict(
                status="passed",
                dynamic_symbols=len(old["symbols"]),
                needed=old["needed"],
                soname=old["soname"],
            ),
            indent=2,
        )
    )
