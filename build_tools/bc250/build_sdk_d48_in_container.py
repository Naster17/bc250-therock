#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compile one modified SDK unit and relink privately from retained inputs."""
import hashlib
import json
from pathlib import Path
import subprocess
from audit_elf_d46 import elf

ROOT = Path("/workspace/d48")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    recipe = json.loads((ROOT / "sdk-recipe.json").read_text())
    compile_args = recipe["compile_argv"].copy()
    link_args = recipe["link_argv"].copy()
    old_object = recipe["agent_object"]
    old_source = "/workspace/source/rocm-systems/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/agent.cpp"
    assert compile_args.count(old_object) == compile_args.count(old_source) == 1
    assert "-DROCPROFILER_CI=1" in compile_args
    obj = ROOT / "candidate-agent.o"
    library = ROOT / "librocprofiler-sdk.so.1.4.2"
    assert not obj.exists() and not library.exists()
    compile_args[compile_args.index(old_object)] = str(obj)
    compile_args[compile_args.index(old_source)] = str(ROOT / "agent.cpp")
    assert link_args.count(old_object) == 1
    link_args[link_args.index(old_object)] = str(obj)
    output = link_args.index("-o") + 1
    assert link_args[output] == "lib/librocprofiler-sdk.so.1.4.2"
    link_args[output] = str(library)
    dependencies = [
        i for i, value in enumerate(link_args) if value.startswith("--dependency-file=")
    ]
    assert len(dependencies) == 1
    link_args[dependencies[0]] = "--dependency-file=" + str(ROOT / "candidate-link.d")
    cwd = "/workspace/build/profiler/rocprofiler-sdk/build"
    record = dict(
        status="building",
        compile_argv=["ccache", *compile_args],
        link_argv=link_args,
        cwd=cwd,
        source_sha256=sha(ROOT / "agent.cpp"),
        ci_preserved=True,
    )
    (ROOT / "commands.json").write_text(json.dumps(record, indent=2) + "\n")
    subprocess.run(["ccache", "--show-stats"], check=True)
    subprocess.run(["ccache", *compile_args], cwd=cwd, check=True)
    subprocess.run(link_args, cwd=cwd, check=True)
    subprocess.run(["ccache", "--show-stats"], check=True)
    raw = library.read_bytes()
    info = elf(raw)
    target = json.loads((ROOT / "runtime-rpath.json").read_text())["value"]
    assert len(info["runtime_paths"]) == 1
    old = info["runtime_paths"][0]
    assert len(target.encode()) <= len(old["value"].encode())
    patched = bytearray(raw)
    start, length = old["offset"], len(old["value"].encode()) + 1
    patched[start : start + length] = target.encode() + b"\0" * (
        length - len(target.encode())
    )
    (ROOT / "librocprofiler-sdk-unrelocated.so.1.4.2").write_bytes(raw)
    library.write_bytes(patched)
    final = elf(patched)
    assert final["runtime_paths"][0]["value"] == target
    assert final["sections"][".text"] == info["sections"][".text"]
    record.update(
        status="built-unqualified",
        library_sha256=sha(library),
        library_bytes=library.stat().st_size,
        object_sha256=sha(obj),
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        rpath=target,
    )
    (ROOT / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {k: record[k] for k in ("status", "library_sha256", "library_bytes")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
