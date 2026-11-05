#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Offline DCP payload exchange for classic Trainer checkpoints on separate hosts.

Export each host, merge on a trusted controller, install missing payload files on
both. Local HF Trainer state/scheduler/RNG files are deliberately not exchanged.
No checkpoint may be in use while exporting/installing. Never unpickle payloads.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import uuid

MAX_PAYLOAD = 256 * 1024**2  # bounded small-checkpoint qualification, not streaming
ROOTS = {"pytorch_model_fsdp_0", "optimizer_0"}


def allowed(name):
    p = PurePosixPath(name)
    return (
        len(p.parts) == 2
        and p.parts[0] in ROOTS
        and p.parts[1] not in [".", ".."]
        and (p.parts[1] == ".metadata" or p.parts[1].endswith(".distcp"))
    )


def pack(values, output):
    assert (
        sum(map(len, values.values())) <= MAX_PAYLOAD
    ), "Payload exceeds 256MiB bounded utility"
    assert not output.exists(), output
    manifest = {
        n: dict(bytes=len(b), sha256=hashlib.sha256(b).hexdigest())
        for n, b in values.items()
    }
    with tarfile.open(output, "x:gz") as t:
        for name, value in [
            *sorted(values.items()),
            (
                "payload-manifest.json",
                (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode(),
            ),
        ]:
            info = tarfile.TarInfo(name)
            info.size = len(value)
            info.mode = 0o644
            t.addfile(info, io.BytesIO(value))


def read(archive):
    values = {}
    with tarfile.open(archive) as t:
        members = t.getmembers()
        assert (
            sum(m.size for m in members) <= MAX_PAYLOAD + 1024**2
        ), "Archive exceeds bounded utility"
        assert len({m.name for m in members}) == len(members), "Duplicate archive names"
        for item in members:
            assert item.isfile() and (
                allowed(item.name) or item.name == "payload-manifest.json"
            ), item.name
            values[item.name] = t.extractfile(item).read()
    manifest = json.loads(values.pop("payload-manifest.json"))
    assert manifest == {
        n: dict(bytes=len(b), sha256=hashlib.sha256(b).hexdigest())
        for n, b in values.items()
    }
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    e = sub.add_parser("export")
    e.add_argument("--checkpoint", type=Path, required=True)
    e.add_argument("--output", type=Path, required=True)
    m = sub.add_parser("merge")
    m.add_argument("--input", type=Path, nargs="+", required=True)
    m.add_argument("--output", type=Path, required=True)
    i = sub.add_parser("install")
    i.add_argument("--checkpoint", type=Path, required=True)
    i.add_argument("--archive", type=Path, required=True)
    a = p.parse_args()
    if a.action == "export":
        values = {}
        for root in sorted(ROOTS):
            directory = a.checkpoint / root
            assert directory.is_dir() and not directory.is_symlink()
            for file in directory.iterdir():
                name = root + "/" + file.name
                assert file.is_file() and not file.is_symlink() and allowed(name), file
                assert (
                    sum(map(len, values.values())) + file.stat().st_size <= MAX_PAYLOAD
                ), "Payload exceeds 256MiB bounded utility"
                values[name] = file.read_bytes()
        assert values
        pack(values, a.output)
    elif a.action == "merge":
        assert (
            len(a.input) == 2 and a.input[0].resolve() != a.input[1].resolve()
        ), "Two distinct host archives required"
        values = {}
        for path in a.input:
            for name, data in read(path).items():
                assert name not in values or values[name] == data, (
                    "Conflicting checkpoint payload",
                    name,
                )
                values[name] = data
        for root in ROOTS:
            assert root + "/.metadata" in values and any(
                n.startswith(root + "/") and n.endswith(".distcp") for n in values
            )
            assert any(
                n.startswith(root + "/__0_") and n.endswith(".distcp") for n in values
            )
            assert any(
                n.startswith(root + "/__1_") and n.endswith(".distcp") for n in values
            )
        pack(values, a.output)
    else:
        assert a.checkpoint.is_dir() and not a.checkpoint.is_symlink()
        values = read(a.archive)
        # Validate EVERY existing destination before writing any missing file.
        for name, data in values.items():
            path = a.checkpoint / name
            assert (
                path.parent.is_dir()
                and not path.parent.is_symlink()
                and not path.is_symlink()
            )
            assert not path.exists() or path.read_bytes() == data, (
                "Refuse overwrite",
                name,
            )
        for name, data in values.items():
            path = a.checkpoint / name
            if not path.exists():
                temp = path.with_name(path.name + ".bc250-" + uuid.uuid4().hex)
                with temp.open("xb") as f:
                    f.write(data)
                    f.flush()
                temp.replace(path)
        assert all((a.checkpoint / n).read_bytes() == b for n, b in values.items())
    print(
        json.dumps(
            dict(
                action=a.action,
                files=len(values),
                payload_bytes=sum(map(len, values.values())),
            )
        )
    )


if __name__ == "__main__":
    main()
