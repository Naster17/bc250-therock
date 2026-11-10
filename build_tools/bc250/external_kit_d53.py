#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Stream and verify a manifest-bound external native build input snapshot."""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tarfile


def digest(stream):
    return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(roots):
    entries = []
    for label, root in roots.items():
        assert label and "/" not in label and label not in (".", "..")
        root = Path(root)
        paths = [root] + sorted(root.rglob("*")) if root.is_dir() else [root]
        for path in paths:
            info = path.lstat()
            name = str(PurePosixPath(label) / path.relative_to(root))
            row = dict(path=name, mode=stat.S_IMODE(info.st_mode))
            if path.is_symlink():
                row.update(kind="symlink", target=os.readlink(path))
            elif path.is_dir():
                row.update(kind="directory")
            elif path.is_file():
                with path.open("rb") as stream:
                    row.update(kind="file", bytes=info.st_size, sha256=digest(stream))
            else:
                raise ValueError(f"Unsupported input: {path}")
            entries.append(row)
    return dict(schema=1, roots={k: str(v) for k, v in roots.items()}, entries=entries)


def create(roots, output):
    manifest = inventory(roots)
    # Do not follow or coalesce links; each file has its own content proof.
    with tarfile.open(
        fileobj=output, mode="w|gz", compresslevel=1, dereference=False
    ) as archive:
        data = json.dumps(manifest, sort_keys=True).encode()
        info = tarfile.TarInfo("manifest.json")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
        for row in manifest["entries"]:
            name = row["path"]
            label, _, relative = name.partition("/")
            path = Path(roots[label]) / relative
            info = archive.gettarinfo(str(path), arcname=name)
            # gettarinfo normally coalesces hardlinks; reset this optimization.
            archive.inodes.clear()
            info.mode = stat.S_IMODE(info.mode)
            assert info.mode == row["mode"]
            if row["kind"] == "file":
                assert info.isfile() and info.size == row["bytes"]
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
            else:
                assert (info.isdir() and row["kind"] == "directory") or (
                    info.issym()
                    and row["kind"] == "symlink"
                    and info.linkname == row["target"]
                )
                archive.addfile(info)
    # Detect ordinary input mutations while exporting. Consumer independently hashes tar bytes.
    assert inventory(roots) == manifest, "Inputs changed during export"
    return manifest


def verify(path):
    with gzip.open(path, "rb") as compressed:
        with tarfile.open(fileobj=compressed, mode="r|") as archive:
            members = iter(archive)
            first = next(members)
            assert (
                first.name == "manifest.json"
                and first.isfile()
                and first.size < 64 * 1024**2
            )
            manifest = json.load(archive.extractfile(first))
            assert manifest["schema"] == 1
            rows = manifest["entries"]
            expected = {r["path"]: r for r in rows}
            assert len(expected) == len(rows)
            seen = set()
            external_links = []
            total = 0
            for member in members:
                name = member.name
                assert name in expected and name not in seen, name
                parts = PurePosixPath(name)
                assert not parts.is_absolute() and ".." not in parts.parts
                row = expected[name]
                assert member.mode == row["mode"]
                if row["kind"] == "file":
                    assert member.isfile() and member.size == row["bytes"]
                    assert digest(archive.extractfile(member)) == row["sha256"], name
                    total += member.size
                elif row["kind"] == "directory":
                    assert member.isdir()
                else:
                    assert row["kind"] == "symlink" and member.issym()
                    assert member.linkname == row["target"]
                    target = os.path.normpath(str(parts.parent / member.linkname))
                    if (
                        target.startswith("/")
                        or target == ".."
                        or target.startswith("../")
                    ):
                        external_links.append(dict(path=name, target=member.linkname))
                    elif target not in expected:
                        external_links.append(dict(path=name, target=member.linkname))
                seen.add(name)
            assert seen == set(expected), set(expected) - seen
        # Drain gzip to enforce its CRC/trailer, including bytes past tar end.
        while tail := compressed.read(1024**2):
            assert not tail.strip(b"\0"), "Non-padding data after tar end"
    with Path(path).open("rb") as stream:
        checksum = digest(stream)
    return dict(
        status="passed",
        entries=len(seen),
        file_bytes=total,
        archive_bytes=Path(path).stat().st_size,
        sha256=checksum,
        external_or_unresolved_links=external_links,
        manifest=manifest,
        scope="Current retained payload bytes; not historical identity, ELF closure, license clearance or clean-build qualification",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["create", "verify"])
    parser.add_argument("--roots", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "create":
        result = create(json.loads(args.roots.read_text()), sys.stdout.buffer)
    else:
        result = verify(args.archive)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
