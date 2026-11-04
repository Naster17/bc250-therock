#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Record completed D04 scope under the same lock used by distributed launches."""
import fcntl
import hashlib
import json
import os
from pathlib import Path


def main():
    assert os.geteuid() == 0
    registry = Path("/opt/bc250-stack/distributed/d01.json")
    record = json.loads(registry.read_text())
    with Path(record["lock"]).open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = Path("/home/nik/bc250work/distributed-d04-fsdp-r1/final.json")
        result = json.loads(report.read_text())
        assert result["status"] == "passed" and result["kernel_log_unchanged"]
        rank = result["rank_result"]
        assert rank["status"] == "passed" and len(rank["checks"]) == 7
        runtime = Path(record["release"]) / "runtime.json"
        assert (
            hashlib.sha256(runtime.read_bytes()).hexdigest() == record["runtime_sha256"]
        )
        record["scope"] = (
            "bounded two-host four-dtype sums, three-step CPU-reference DDP/SGD, "
            "nested FSDP/AdamW parameter/state shards and same-world checkpoint resume; "
            "training scaling, large-model capacity and other protocols unqualified"
        )
        record["D04_report"] = str(report)
        replacement = registry.with_suffix(".d04-new")
        replacement.write_text(json.dumps(record, indent=2) + "\n")
        replacement.chmod(0o444)
        replacement.replace(registry)


if __name__ == "__main__":
    main()
