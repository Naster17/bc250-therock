#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""D35 training/reference identity with an explicit D36 audit implementation."""
import argparse
from pathlib import Path
import sys
import distributed_d35_growth as base
from audit_units_d36 import Auditor
from artifact_d35 import source_hashes, digest


def sources():
    result = source_hashes()
    result.update(
        {
            name: digest(Path(__file__).with_name(name))
            for name in [
                "distributed_d36_units.py",
                "audit_units_d36.py",
                "unit_plan_d36.py",
            ]
        }
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--audit", choices=["whole", "unit"], required=True)
    args, remaining = parser.parse_known_args()
    assert remaining[remaining.index("--mode") + 1] == "fsdp"
    assert remaining[remaining.index("--phase") + 1] == "qualify"
    scratch = (
        Path(remaining[remaining.index("--output") + 1]).parent / "actual-gradients"
    )
    auditor = Auditor(args.audit, scratch)
    auditor.install_guard()
    base.qualify = auditor.qualify
    base.sources = sources
    sys.argv = [sys.argv[0], *remaining]
    base.main()


if __name__ == "__main__":
    main()
