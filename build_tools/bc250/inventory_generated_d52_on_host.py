#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Read retained compile inputs and declared native link targets without build tools."""
import hashlib
import json
from pathlib import Path
import re
import shlex

WORK = Path("/home/nik/bc250work")
SOURCE = WORK / "tools/source-d51/build-inputs-r1/evidence"
ROOT = WORK / "evidence/source-d52-inputs-r2"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def native_link_rules(ninja):
    return [
        line
        for line in ninja.splitlines()
        if line.startswith("build ")
        and ": phony" not in line
        and re.search(r"\.so(?:\.[0-9.]+)?(?:\s|$)|\.a(?:\s|$)", line.split(":", 1)[0])
    ]


def main():
    ROOT.mkdir(exist_ok=False)
    report = dict(
        status="passed",
        profiles={},
        scope="Retained compilation-unit bytes, include-directory existence and declared link rules; not full transitive header/.ninja_deps/link-ELF closure",
    )
    cache = {}
    for profile in ("daily", "rccl-single"):
        root = SOURCE / profile
        command = json.loads((root / "retained-build.json").read_text())["command"]
        mounts = []
        for i, value in enumerate(command):
            if value == "-v":
                host, container, mode = command[i + 1].rsplit(":", 2)
                mounts.append((Path(container), Path(host)))

        def mapped(value):
            path = Path(value)
            candidates = [
                (container, host)
                for container, host in mounts
                if path == container or container in path.parents
            ]
            if not candidates:
                return None
            container, host = max(candidates, key=lambda item: len(item[0].parts))
            return host / path.relative_to(container)

        commands = json.loads((root / "compile_commands.json").read_text())
        files = []
        includes = {}
        for row in commands:
            path = Path(row["file"])
            path = path if path.is_absolute() else Path(row["directory"]) / path
            host = mapped(path)
            record = dict(
                file=str(path),
                host_path=str(host) if host is not None else None,
                under_build_directory="/build/" in str(path),
            )
            if host is not None and host.is_file():
                key = str(host.resolve())
                if key not in cache:
                    cache[key] = sha(host)
                record.update(
                    present=True, bytes=host.stat().st_size, sha256=cache[key]
                )
            else:
                record["present"] = False
            files.append(record)
            argv = row.get("arguments") or shlex.split(row["command"])
            for i, token in enumerate(argv):
                value = (
                    argv[i + 1]
                    if token in ("-I", "-isystem", "-iquote") and i + 1 < len(argv)
                    else token[2:] if token.startswith("-I") else None
                )
                if value and value not in includes:
                    p = mapped(value)
                    includes[value] = dict(
                        host_path=str(p) if p else None,
                        present=p.is_dir() if p else None,
                    )
        ninja = (root / "build.ninja").read_text()
        links = native_link_rules(ninja)
        report["profiles"][profile] = dict(
            compile_commands_sha256=sha(root / "compile_commands.json"),
            ninja_sha256=sha(root / "build.ninja"),
            units=len(files),
            present_units=sum(x["present"] for x in files),
            under_build_units=sum(x["under_build_directory"] for x in files),
            unresolved_units=[x["file"] for x in files if not x["present"]],
            files=files,
            include_directories=includes,
            declared_native_link_rules=links,
        )
    (ROOT / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: {
                    k: value[k]
                    for k in [
                        "units",
                        "present_units",
                        "under_build_units",
                        "unresolved_units",
                    ]
                }
                for key, value in report["profiles"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
