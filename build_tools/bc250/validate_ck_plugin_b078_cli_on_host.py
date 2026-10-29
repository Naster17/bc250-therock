#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exercise public host check/seed commands under the independent runtime guard."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    r = dict(status="running", checks=[])
    try:
        for profile in ("daily", "rccl-single"):
            base = ["/usr/local/bin/bc250-ck", "--runtime-profile", profile]
            for label, tail, code in [
                ("check", ["--check"], 0),
                ("seed", ["--seed-db", str(args.output.parent / ("db-" + profile))], 0),
                (
                    "existing",
                    ["--seed-db", str(args.output.parent / ("db-" + profile))],
                    2,
                ),
            ]:
                proc = subprocess.run(
                    [*base, *tail],
                    capture_output=True,
                    text=True,
                    stdin=subprocess.DEVNULL,
                    timeout=90,
                )
                (args.output.parent / (profile + "-" + label + ".log")).write_text(
                    proc.stdout + proc.stderr
                )
                assert proc.returncode == code, (label, proc.stderr)
                if label == "check":
                    assert json.loads(proc.stdout)["status"] == "passed"
                if label == "seed":
                    root = args.output.parent / ("db-" + profile)
                    before = {
                        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in root.glob("*.txt")
                    }
                    assert len(before) == 2
                if label == "existing":
                    assert before == {
                        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in root.glob("*.txt")
                    }
                r["checks"].append(profile + "-" + label)
        r["status"] = "passed"
    except BaseException as error:
        r.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(r, indent=2) + "\n")


if __name__ == "__main__":
    main()
