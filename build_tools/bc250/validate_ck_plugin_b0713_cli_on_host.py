#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exercise public host check/ck-info/seed commands under the independent runtime guard.

8 checks: for each of daily and rccl-single, --check (passed JSON, 27 mapped
runtime identities, private MIOpen from the installed root), --ck-info (exact
b0712 registration pins), --seed-db (two .txt files, each with 6 FP16 + 2 FP32
keys) and the existing-destination refusal leaving the seeded bytes unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from bc250_ck_b0713 import ARCHIVE_SHA, BASE, MANIFEST_SHA, NAME

ROOT = BASE / NAME


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
                ("ck-info", ["--ck-info"], 0),
                (
                    "seed",
                    ["--seed-db", str(args.output.parent / ("db-" + profile))],
                    0,
                ),
                (
                    "existing",
                    ["--seed-db", str(args.output.parent / ("db-" + profile))],
                    2,
                ),
            ]:
                # --ck-info is a profile-agnostic sole-argument command; the
                # wrapper rejects launcher options before it, so invoke it bare.
                cmd = (
                    ["/usr/local/bin/bc250-ck", *tail]
                    if label == "ck-info"
                    else [*base, *tail]
                )
                proc = subprocess.run(
                    cmd,
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
                    value = json.loads(proc.stdout.strip().splitlines()[-1])
                    assert value["status"] == "passed"
                    assert value["artifact"] == NAME
                    assert value["architecture"] == "gfx1013:xnack-"
                    assert value["api"] == 5
                    assert len(value["runtime_libraries"]) == 27
                    assert value["runtime_libraries"]["libMIOpen.so.1"]["path"] == str(
                        ROOT / "runtime/libMIOpen.so.1"
                    )
                if label == "ck-info":
                    record = json.loads(proc.stdout)
                    assert record["package"] == str(ROOT)
                    assert record["archive_sha256"] == ARCHIVE_SHA
                    assert record["manifest_sha256"] == MANIFEST_SHA
                if label == "seed":
                    root = args.output.parent / ("db-" + profile)
                    before = {
                        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in root.glob("*.txt")
                    }
                    assert len(before) == 2
                    for file in root.glob("*.txt"):
                        lines = file.read_text().splitlines()
                        assert (
                            len(lines) == 8
                            and sum("FP16" in x for x in lines) == 6
                            and sum("FP32" in x for x in lines) == 2
                        )
                if label == "existing":
                    assert before == {
                        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in root.glob("*.txt")
                    }
                r["checks"].append(profile + "-" + label)
        assert len(r["checks"]) == 8
        r["status"] = "passed"
    except BaseException as error:
        r.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(r, indent=2) + "\n")


if __name__ == "__main__":
    main()
