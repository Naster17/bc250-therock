#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Read-only ELF64 section/RPATH comparison; never loads or executes a DSO."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def sha(data):
    return hashlib.sha256(data).hexdigest()


def elf(data):
    if len(data) < 64 or data[:6] != b"\x7fELF\x02\x01":
        raise ValueError("Expected little-endian ELF64")
    header = struct.unpack_from("<16sHHIQQQIHHHHHH", data)
    offset, size, count, strings = header[6], header[11], header[12], header[13]
    if (
        size != 64
        or count == 0
        or strings >= count
        or offset + size * count > len(data)
    ):
        raise ValueError("Invalid/unsupported section table")
    entries = [
        struct.unpack_from("<IIQQQQIIQQ", data, offset + i * size) for i in range(count)
    ]

    def payload(item):
        start, length = item[4:6]
        if start + length > len(data):
            raise ValueError("Truncated section")
        return data[start : start + length]

    names = payload(entries[strings])

    def string(table, index):
        if index >= len(table) or table.find(b"\0", index) < 0:
            raise ValueError("Invalid string offset")
        return table[index : table.index(b"\0", index)].decode("utf-8")

    sections = {}
    for item in entries:
        name = string(names, item[0])
        if item[1] != 8:  # SHT_NOBITS is not file-backed.
            sections[name] = dict(
                offset=item[4], size=item[5], flags=item[2], sha256=sha(payload(item))
            )
    paths = []
    if ".dynamic" in sections and ".dynstr" in sections:
        dynamic = sections[".dynamic"]
        table = sections[".dynstr"]
        strings_data = data[table["offset"] : table["offset"] + table["size"]]
        if dynamic["size"] % 16:
            raise ValueError("Invalid dynamic table")
        for pos in range(dynamic["offset"], dynamic["offset"] + dynamic["size"], 16):
            tag, value = struct.unpack_from("<qQ", data, pos)
            if tag in (15, 29):
                value_string = string(strings_data, value)
                paths.append(
                    dict(tag=tag, value=value_string, offset=table["offset"] + value)
                )
    return dict(
        sha256=sha(data), bytes=len(data), sections=sections, runtime_paths=paths
    )


def compare(build, runtime):
    left, right = elf(build), elf(runtime)
    changed = [
        name
        for name in sorted(set(left["sections"]) | set(right["sections"]))
        if left["sections"].get(name) != right["sections"].get(name)
    ]
    replay = bytearray(build)
    transformations = []
    for source, target in zip(left["runtime_paths"], right["runtime_paths"]):
        old = source["value"].encode()
        new = target["value"].encode()
        if (
            source["tag"] != target["tag"]
            or source["offset"] != target["offset"]
            or len(new) > len(old)
        ):
            continue
        start = source["offset"]
        replay[start : start + len(old) + 1] = new + b"\0" * (len(old) + 1 - len(new))
        transformations.append(
            dict(old=source["value"], new=target["value"], offset=start)
        )
    exact = bool(transformations) and bytes(replay) == runtime
    return dict(
        build=left,
        runtime=right,
        changed_sections=changed,
        rpath_transformations=transformations,
        rpath_only_exact_replay=exact,
        replay_sha256=sha(replay),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            compare(args.build.read_bytes(), args.runtime.read_bytes()), indent=2
        )
    )
