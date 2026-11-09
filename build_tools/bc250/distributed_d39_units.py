#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""D35 training/reference identity with an explicit D39 clipping audit implementation."""
import argparse
from pathlib import Path
import sys
import distributed_d35_growth as base
from audit_units_d39 import Auditor
from artifact_d35 import source_hashes, digest, admit_storage


def sources():
    result = source_hashes()
    result.update(
        {
            name: digest(Path(__file__).with_name(name))
            for name in [
                "distributed_d36_units.py",
                "audit_units_d36.py",
                "unit_plan_d36.py",
                "distributed_d39_units.py",
                "audit_units_d39.py",
            ]
        }
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--audit", choices=["legacy", "reference"], required=True)
    args, remaining = parser.parse_known_args()
    assert remaining[remaining.index("--mode") + 1] == "fsdp"
    assert remaining[remaining.index("--phase") + 1] == "qualify"
    scratch = (
        Path(remaining[remaining.index("--output") + 1]).parent / "actual-gradients"
    )
    auditor = Auditor(args.audit, scratch)
    auditor.install_guard()
    if args.audit == "reference":

        def storage(free, parameters, *, producing):
            assert parameters == base.PARAMETERS and producing is False
            return admit_storage(free, 0, producing=False)

        base.admit_storage = storage
        native_atomic_json = base.atomic_json

        def publish(path, record):
            if record["status"] == "passed":
                record["checks"][
                    2
                ] = "CPU-root-norm-independent-FP64-reference-ALL-clipped-gradients"
            native_atomic_json(path, record)

        base.atomic_json = publish
    base.qualify = auditor.qualify
    base.sources = sources
    sys.argv = [sys.argv[0], *remaining]
    base.main()


if __name__ == "__main__":
    main()
