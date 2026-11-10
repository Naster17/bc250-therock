#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Package/verify the assembled working-tree bytes, excluding private Git metadata."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile

PREFIX = "torch-source-d52-r1"


def digest(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        value.update(block)
    return value.hexdigest()


def record(path):
    if path.is_symlink():
        return dict(type="symlink", target=os.readlink(path))
    with path.open("rb") as stream:
        value = digest(stream)
    return dict(
        type="file",
        bytes=path.stat().st_size,
        executable=bool(path.stat().st_mode & 0o100),
        sha256=value,
    )


def create(assembly, archive):
    proof = json.loads((assembly / "assembly.json").read_text())
    assert proof["status"] == "passed"
    assert not archive.exists()
    files = {}
    for base in ["torch", "recipes"]:
        for current, dirs, names in os.walk(assembly / base, followlinks=False):
            dirs[:] = [name for name in dirs if name != ".git"]
            for name in list(dirs):
                p = Path(current) / name
                if p.is_symlink():
                    files[str(p.relative_to(assembly))] = record(p)
                    dirs.remove(name)
            for name in names:
                p = Path(current) / name
                files[str(p.relative_to(assembly))] = record(p)
    sourcefiles = sum(name.startswith("torch/") for name in files)
    assert (
        sourcefiles
        == proof["parent_blob_files"]
        + proof["dependency_blob_files_with_duplicate_locations"]
    )
    manifest = assembly / "files-d52.json"
    manifest.write_text(
        json.dumps(
            dict(source_files=sourcefiles, files=files),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    (assembly / "README-D52.md").write_text(
        """# D52 assembled selected source snapshot\n\nThe torch/ directory contains verified parent source8d8b18d on native C++ basea524980 plus69 dependency locations and the recorded Gloo overlay. Eight excluded nested branches are empty. Gitless archive: private verification indices/histories remain outside this package.\n\nrecipes/build-fresh-{daily,rccl-single}.sh are prospective pip-only recipes, syntax checked but not executed here. They refuse a pre-existing CMake cache; the RCCL recipe requires /workspace/rccl. The original historical cache/graphs/recipes under recipes/{daily,rccl-single} are evidence, not a starting build cache. Do not copy those caches into torch/build.\n\nRequires separately preserved native builder28ad36e6, matching external ROCm SDK/compiler/dependencies and admitted writable cache/output storage. Image export and source reconstruction do not establish a clean native build or complete licenses/offline build closure. Use explicit build-version/number1 and nativegfx1013. Native installed Torch git_version remains Unknown; assembly.json carries actual source provenance. No original source/prefix/default was changed to assemble this snapshot.\n\nVerify the archive with source_kit_d52.py verify. For extraction, use its extract command, which preserves trailing slash symlink spelling after standard data-filter boundary checks.\n"""
    )
    with tarfile.open(archive, "w:gz", dereference=False) as output:
        output.add(manifest, arcname=PREFIX + "/files-d52.json")
        for name in ["assembly.json", "README-D52.md"]:
            output.add(assembly / name, arcname=PREFIX + "/" + name)
        for base in ["torch", "recipes"]:

            def filter_member(member):
                return None if ".git" in Path(member.name).parts else member

            output.add(
                assembly / base, arcname=PREFIX + "/" + base, filter=filter_member
            )
    return verify(archive)


def verify(archive):
    with tarfile.open(archive, "r:gz") as content:
        first = content.next()
        assert first.name == PREFIX + "/files-d52.json"
        manifest = json.loads(content.extractfile(first).read())
        want = manifest["files"]
        seen = set()
        for member in content:
            parts = Path(member.name).parts
            assert (
                parts
                and parts[0] == PREFIX
                and ".." not in parts
                and ".git" not in parts
            )
            relative = str(Path(*parts[1:]))
            if relative in ("files-d52.json", "assembly.json", "README-D52.md"):
                continue
            if member.isdir():
                continue
            assert relative not in seen
            seen.add(relative)
            expected = want[relative]
            if member.issym():
                assert expected == dict(type="symlink", target=member.linkname)
            else:
                assert member.isfile() and expected["type"] == "file"
                assert (
                    member.size == expected["bytes"]
                    and bool(member.mode & 0o100) == expected["executable"]
                )
                assert (
                    digest(content.extractfile(member)) == expected["sha256"]
                ), member.name
        assert seen == set(want)
    with archive.open("rb") as stream:
        checksum = digest(stream)
    return dict(
        status="passed",
        archive_sha256=checksum,
        archive_bytes=archive.stat().st_size,
        source_files=manifest["source_files"],
        all_recipe_and_source_entries=len(seen),
        private_git_metadata_bundled=False,
        scope="Recorded exact working-tree bytes/modes/symlinks; no configure/native build or full external SDK/license closure",
    )


def extract(archive, destination):
    result = verify(archive)
    destination.mkdir(parents=True, exist_ok=False)

    def exact_filter(member, path):
        checked = tarfile.data_filter(member, path)
        if checked is not None and member.issym():
            assert member.linkname.rstrip("/") == checked.linkname.rstrip("/")
            checked = checked.replace(linkname=member.linkname)
        return checked

    with tarfile.open(archive) as content:
        content.extractall(destination, filter=exact_filter)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["create", "verify", "extract"])
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--assembly", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "create":
        result = create(args.assembly, args.archive)
    elif args.action == "extract":
        result = extract(args.archive, args.destination)
    else:
        result = verify(args.archive)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
