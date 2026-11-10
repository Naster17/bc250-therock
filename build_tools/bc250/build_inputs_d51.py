#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Classify pinned missing branches and parse explicit retained build settings."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex

FAMILIES = ("third_party/aiter", "third_party/flash-attention", "third_party/mslk")
OFF_FLAGS = (
    "USE_FLASH_ATTENTION",
    "USE_MEM_EFF_ATTENTION",
    "USE_ROCM_CK_SDPA",
    "USE_ROCM_CK_GEMM",
    "USE_MSLK",
)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cache_values(text):
    return {
        line.split(":", 1)[0]: line.partition("=")[2]
        for line in text.splitlines()
        if ":" in line and "=" in line and not line.startswith(("#", "//"))
    }


def build_settings(text):
    result = {}
    for line in text.splitlines():
        if line.startswith("export "):
            for token in shlex.split(line[7:]):
                key, sep, value = token.partition("=")
                if sep and (
                    key.startswith(("USE_", "PYTORCH_", "CMAKE_"))
                    or key in ("MAX_JOBS", "BLAS", "CCACHE_MAXSIZE")
                ):
                    result[key] = value
    assert result["PYTORCH_BUILD_VERSION"] == "2.13.0a0+gfx1013"
    assert result["PYTORCH_BUILD_NUMBER"] == "1"
    assert result["PYTORCH_ROCM_ARCH"] == "gfx1013"
    assert re.search(
        r"^python -m pip install -e \. -v --no-build-isolation$", text, re.M
    )
    return result


def classify(cache, commands, missing):
    flags = cache_values(cache)
    assert all(
        flags.get(k, "").upper() in ("0", "OFF", "FALSE") for k in OFF_FLAGS
    ), "Optional backend flags differ; re-audit required"
    references = {
        family: [c["file"] for c in commands if family in json.dumps(c)]
        for family in FAMILIES
    }
    assert all(
        not values for values in references.values()
    ), "Optional branch is referenced by native compile database"
    result = []
    for item in missing:
        assert any(item["path"].startswith(family + "/") for family in FAMILIES)
        result.append(
            dict(
                path=item["path"],
                pin=item["pin"],
                disposition="excluded-from-this-retained-native-configuration",
                evidence="all five optional flags off; no family token in complete retained compile database; source CMake guards separately archived",
                future_configuration="requires fresh dependency/feature admission; descendants remain unexpanded",
            )
        )
    assert len(result) == 8
    return dict(
        compile_entries=len(commands),
        flags={key: flags[key] for key in OFF_FLAGS},
        references=references,
        missing=result,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--commands", type=Path, required=True)
    parser.add_argument("--missing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = classify(
        args.cache.read_text(),
        json.loads(args.commands.read_text()),
        json.loads(args.missing.read_text()),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
