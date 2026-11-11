#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Inspect retained Ninja header records, declared link inputs and ELF resolution."""

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

SOURCE = Path("/workspace/pytorch-work")
BUILD = SOURCE / "build"
SDK = Path("/workspace/build/dist/rocm")
RCCL = Path(os.environ.get("RCCL_ROOT", "/workspace/rccl"))
LIBRARY = re.compile(r"(?:\.a|\.so(?:\.[\w.-]+)?)$")
ALLOWED_LIBRARY_ROOTS = (
    BUILD,
    SOURCE / "torch",
    SDK,
    RCCL,
    Path("/usr"),
    Path("/opt/torch-venv"),
    Path("/opt/build-venv"),
    Path("/usr/local/therock-tools"),
)


def command(args, **kwargs):
    return subprocess.check_output(args, text=True, timeout=120, **kwargs)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def header_records():
    return parse_header_records(command(["ninja", "-t", "deps"], cwd=BUILD))


def parse_header_records(output):
    dependencies = set()
    records = []
    assert output.strip(), "Empty Ninja dependency database"
    for block in output.strip().split("\n\n"):
        rule, *paths = block.splitlines()
        obj, count, state = re.fullmatch(
            r"(.+): #deps (\d+), deps mtime \d+ \((VALID|STALE)\)", rule
        ).groups()
        assert len(paths) == int(count), rule
        dependencies.update(path.strip() for path in paths)
        records.append((obj, state))
    return dependencies, records


def link_inputs():
    inputs = set()
    bare = set()
    for line in (BUILD / "build.ninja").read_text().splitlines():
        if line.startswith("  LINK_LIBRARIES = "):
            tokens = shlex.split(line.split(" = ", 1)[1])
            for token in tokens:
                for argument in token.split(","):
                    if argument.startswith("-l") and not argument.startswith("-L"):
                        bare.add(argument)
                    elif not argument.startswith("-") and LIBRARY.search(argument):
                        inputs.add(argument)
    for flag in bare:
        name = flag[2:]
        shared = command(["g++", f"-print-file-name=lib{name}.so"]).strip()
        static = command(["g++", f"-print-file-name=lib{name}.a"]).strip()
        choices = [p for p in (shared, static) if Path(p).is_absolute()]
        inputs.update(choices or [flag])
    return inputs, bare


def loader_closure(profile):
    env = os.environ.copy()
    roots = [SDK / path for path in ("lib", "lib/llvm/lib", "lib/rocm_sysdeps/lib")]
    roots += [SOURCE / "torch/lib", BUILD / "lib"]
    if profile == "rccl-single":
        roots.append(RCCL / "lib")
    env["LD_LIBRARY_PATH"] = ":".join(map(str, roots))
    objects = sorted((SOURCE / "torch/lib").glob("*.so"))
    objects += list((SOURCE / "torch").glob("_C.*.so"))
    objects.append(SDK / "lib/llvm/bin/clang")
    if profile == "rccl-single":
        objects.append(RCCL / "lib/librccl.so")
    result = {}
    paths = set()
    for obj in objects:
        completed = subprocess.run(
            ["ldd", str(obj)], env=env, text=True, capture_output=True, timeout=30
        )
        resolved = re.findall(r"(?:=>\s+|^\s*)(/[^\s]+)\s+\(", completed.stdout, re.M)
        paths.update(resolved)
        result[str(obj)] = dict(
            dynamic=command(["readelf", "-d", str(obj)]),
            code=completed.returncode,
            output=completed.stdout + completed.stderr,
            unresolved=[
                line.strip()
                for line in completed.stdout.splitlines()
                if "=> not found" in line
            ],
        )
    outside = []
    for value in sorted(paths):
        path = Path(value).resolve()
        if not any(path.is_relative_to(root) for root in ALLOWED_LIBRARY_ROOTS):
            outside.append(value)
    return result, sorted(paths), outside


def main(profile):
    inputs, bare = link_inputs()
    headers, records = header_records()
    loader, providers, outside = loader_closure(profile)
    report = dict(
        profile=profile,
        deps_database_sha256=digest(BUILD / ".ninja_deps"),
        dependency_records=len(records),
        stale_records=sum(state == "STALE" for _, state in records),
        unique_dependencies=len(headers),
        missing_dependencies=sorted(
            path for path in headers if not (BUILD / path).is_file()
        ),
        link_inputs=sorted(inputs),
        bare_link_flags=sorted(bare),
        missing_link_inputs=sorted(
            path for path in inputs if not (BUILD / path).is_file()
        ),
        loader=loader,
        loader_paths=providers,
        unexpected_loader_paths=outside,
    )
    Path("/evidence/retained-audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("loader", "link_inputs", "loader_paths")
            }
        )
    )
    assert not outside and all(
        not r["code"] and not r["unresolved"] for r in loader.values()
    )


if __name__ == "__main__":
    main(sys.argv[1])
