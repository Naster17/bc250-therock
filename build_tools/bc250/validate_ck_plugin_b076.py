#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Qualify native public Search/PerfDb and fresh-process replay in private DBs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess


def hashes(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.glob("*.txt")
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--build-attempt", choices=("r1", "r2", "r3"), required=True)
    args = p.parse_args()
    ws = Path("/home/nik/bc250work/rocm-10.2-gfx1013")
    root = args.output.parent
    db = root / "selected-db"
    db.mkdir()
    cache = root / "private-kernels"
    result = dict(status="running", checks=[], runs=[])
    base = dict(
        os.environ,
        MIOPEN_USER_DB_PATH=str(db),
        MIOPEN_CUSTOM_CACHE_DIR=str(cache),
        MIOPEN_FIND_MODE="NORMAL",
        MIOPEN_TUNING_ITERATIONS="5",
        MIOPEN_TUNING_TIME_MS_MAX="1000",
    )

    def run(label, flags, extra):
        folder = root / label
        folder.mkdir()
        env = dict(base, **extra)
        cmd = [
            str(ws / f"tmp/ck-plugin-b076/build-{args.build_attempt}/probe-b076"),
            *flags,
        ]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=180)
        (folder / "stdout.log").write_text(proc.stdout)
        (folder / "stderr.log").write_text(proc.stderr)
        (folder / "environment.json").write_text(
            json.dumps(
                {k: v for k, v in env.items() if k.startswith("MIOPEN_")}, indent=2
            )
            + "\n"
        )
        proc.check_returncode()
        value = json.loads(proc.stdout.strip().splitlines()[-1])
        assert (
            value["status"] == "passed" and value["guards"] and len(value["rows"]) == 2
        )
        assert value["dl_solver"] == 137 and value["direct_solver"] == 218
        return value, proc.stderr

    try:
        tuned, log = run(
            "native-search",
            ["--tune"],
            dict(
                MIOPEN_LOG_LEVEL="6",
                MIOPEN_FIND_ENFORCE="SEARCH_DB_UPDATE",
                MIOPEN_DEBUG_FIND_ONLY_SOLVER="ConvHipImplicitGemmGroupFwdXdlops",
            ),
        )
        assert log.count("Search Started:") >= 2 and log.count("Search Ended:") >= 2
        result["tuned"] = tuned
        result["checks"].append(
            "public-Find-solver137-native-Search-two-reference-fixtures"
        )
        perfs = list(db.glob("*.udb.txt"))
        finds = list(db.glob("*.ufdb.txt"))
        assert len(perfs) == len(finds) == 1
        records = perfs[0].read_text().splitlines()
        assert len(records) == 2 and all(
            "=ConvHipImplicitGemmGroupFwdXdlops:" in line for line in records
        )
        ids = [
            line.split("=ConvHipImplicitGemmGroupFwdXdlops:", 1)[1] for line in records
        ]
        known = json.loads(
            (ws / "logs/ck-plugin-b075-r1-private-daily/result.json").read_text()
        )["runs"][0]["kernel_ids"]
        assert all(value in known for value in ids)
        assert len(finds[0].read_text().splitlines()) == 2
        snapshot = hashes(db)
        result.update(
            perfdb_records=records, selected_kernel_ids=ids, database_hashes=snapshot
        )
        result["checks"].append(
            "genuine-two-PerfDb-and-two-FindDb-records-no-manual-selection"
        )
        for index in range(2):
            r, log = run(
                "public-paired-" + str(index),
                ["--reverse"] if index else [],
                dict(MIOPEN_LOG_LEVEL="3", MIOPEN_FIND_ENFORCE="NONE"),
            )
            assert not r["tune"] and not r["diagnostic"]
            for row in r["rows"]:
                assert len(row["windows"]) == 12
                row["median_stream_ms"] = [
                    statistics.median(
                        w["stream_interval_ms_per_call"]
                        for w in row["windows"]
                        if w["solver_index"] == j
                    )
                    for j in (0, 1)
                ]
                row["median_wall_ms"] = [
                    statistics.median(
                        w["wall_ms_per_call"]
                        for w in row["windows"]
                        if w["solver_index"] == j
                    )
                    for j in (0, 1)
                ]
                row["direct_over_dl_ratio"] = (
                    row["median_stream_ms"][1] / row["median_stream_ms"][0]
                )
            assert hashes(db) == snapshot, "replay mutated recorded DB"
            result["runs"].append(r)
            result["checks"].append(
                "fresh-process-selected-public-Immediate-CPU-reference-paired-"
                + str(index)
            )
        r, log = run(
            "replay-attribution",
            ["--diagnostic"],
            dict(MIOPEN_LOG_LEVEL="6", MIOPEN_FIND_ENFORCE="NONE"),
        )
        assert "Search Started:" not in log
        assert (
            log.count("Perf Db: record loaded: ConvHipImplicitGemmGroupFwdXdlops") >= 2
        )
        assert all("kernel_name = " + kid in log for kid in ids)
        assert "kernel_name = MIOpenConvUni" in log and hashes(db) == snapshot
        result["diagnostic"] = r
        result["checks"].append(
            "fresh-process-PerfDb-load-exact-ID-attribution-without-search-or-DB-change"
        )
        bad = root / "malformed-db"
        shutil.copytree(db, bad)
        bad_perf = bad / perfs[0].name
        text, count = re.subn(
            r"DeviceGroupedConvFwdDlMultipleD_[^>]+>",
            "invalid-b076-kernel-id",
            bad_perf.read_text(),
        )
        assert count == 2
        bad_perf.write_text(text)
        bad_hashes = hashes(bad)
        r, log = run(
            "malformed-fallback",
            ["--diagnostic"],
            dict(
                MIOPEN_LOG_LEVEL="6",
                MIOPEN_FIND_ENFORCE="NONE",
                MIOPEN_USER_DB_PATH=str(bad),
            ),
        )
        assert (
            log.count(
                "Invalid config loaded from Perf Db: ConvHipImplicitGemmGroupFwdXdlops"
            )
            >= 2
        )
        assert (
            "kernel_name = DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK<8, 16, 4, 2, Default, 1>"
            in log
        )
        assert (
            "Search Started:" not in log
            and hashes(bad) == bad_hashes
            and hashes(db) == snapshot
        )
        result["malformed_fallback"] = r
        result["checks"].append(
            "invalid-IDs-refused-default-public-reference-fallback-original-DB-preserved"
        )
        import torch

        x = torch.arange(35, device="cuda", dtype=torch.float32).requires_grad_()
        x.square().sum().backward()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            x.grad.cpu(), 2 * torch.arange(35, dtype=torch.float32), rtol=0, atol=0
        )
        result["checks"].append("native-torch-backward-after-native-public-tuning")
        assert len(result["checks"]) == 7
        result["status"] = "passed"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
