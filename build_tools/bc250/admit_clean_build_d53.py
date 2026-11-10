#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Conservative proposed sequential native build disk admission; never starts builds."""
import argparse
import json
from pathlib import Path
import shutil

GIB = 1024**3
# Proposed allocation ceilings, not measured clean-build peak requirements.
BUDGET = dict(
    fresh_source_and_hipify=3 * GIB,
    native_objects_and_codegen=8 * GIB,
    link_and_install_scratch=4 * GIB,
    isolated_ccache_growth=20 * GIB,
    preserved_free_reserve=5 * GIB,
)


def admission(free_bytes, budget=None):
    budget = BUDGET if budget is None else budget
    assert free_bytes >= 0 and all(x >= 0 for x in budget.values())
    required = sum(budget.values())
    return dict(
        admitted=free_bytes >= required,
        free_bytes=free_bytes,
        required_bytes=required,
        shortfall_bytes=max(0, required - free_bytes),
        proposed_budget=budget,
        scope="Sequential one-profile proposal reusing immutable read-only SDK; no extraction duplicate. Not a measured sufficient peak or permission to launch. Memory/ELF/license/recipe closure requires separate gates.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = admission(shutil.disk_usage(args.path).free)
    report["path"] = str(args.path.resolve())
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
