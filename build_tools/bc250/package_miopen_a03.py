#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Package only genuine A03 native records after guarded GPU/reference collection."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def gate(ws, label, count):
    p = ws / "logs" / label
    assert json.loads((p / "exit.json").read_text()) == dict(
        returncode=0, kernel_log_unchanged=True
    )
    assert "GUARD_ACKNOWLEDGED" in (p / "guard.log").read_text()
    r = json.loads((p / "result.json").read_text())
    assert r["status"] == "passed" and len(r["results"]) == count
    assert r["arch"].split(":")[0] == "gfx1013" and r["multiprocessors"] == 20
    assert all(x["status"] == "passed" for x in r["results"])
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--workspace", type=Path, default=Path("/home/nik/bc250work/rocm-10.2-gfx1013")
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--source-manifest", type=Path, required=True)
    args = ap.parse_args()
    ws = args.workspace
    collection = gate(ws, "miopen-a03-collect", 15)
    tune = gate(ws, "miopen-a03-tune", 1)
    normal = gate(ws, "miopen-a03-tune-normal", 1)
    assert tune["environment"]["MIOPEN_FIND_ENFORCE"] == "SEARCH_DB_UPDATE"
    assert tune["environment"]["MIOPEN_DEBUG_FIND_ONLY_SOLVER"] == "ConvHipDirectFwd"
    source = json.loads(args.source_manifest.read_text())
    for rel, expected in source["sources_sha256"].items():
        assert sha(ws / "source" / rel) == expected
    root = ws / "cache/a03-collect"
    fdb = root / "db/gfx1013_20.HIP.3_6_2_.ufdb.txt"
    perf = root / "db/gfx1013_20.HIP.3_6_2_.udb.txt"
    kernels = root / "kernels/3.6.2./gfx1013_20.ukdb"
    records = fdb.read_text().splitlines()
    assert len(records) == 16 and len({x.split("=", 1)[0] for x in records}) == 16
    solutions = sum(
        len(x["solutions"]) for x in collection["results"] + normal["results"]
    )
    assert sum(len(x.split("=", 1)[1].split(";")) for x in records) == solutions
    perf_records = perf.read_text().splitlines()
    assert len(perf_records) == 1 and "=ConvHipDirectFwd:" in perf_records[0]
    assert not args.output.exists(), "Keep prior packages and evidence"
    db = args.output / "db"
    db.mkdir(parents=True)
    shutil.copy2(fdb, db / "gfx1013_20.HIP.fdb.txt")
    shutil.copy2(perf, db / "gfx1013_20.db.txt")
    # Identical user/system kernel DB schemas. SQLite backup takes a consistent
    # snapshot without changing or moving the original cache (including WALs).
    with sqlite3.connect("file:" + str(kernels) + "?mode=ro", uri=True) as c:
        assert c.execute("pragma integrity_check").fetchone() == ("ok",)
        rows = c.execute(
            "select kernel_name,kernel_args,kernel_hash,uncompressed_size from kern_db"
        ).fetchall()
        assert rows and all("gfx1013" in x[1] for x in rows)
        with sqlite3.connect(db / "gfx1013_20.kdb") as target:
            c.backup(target)
            # User caches use WAL; system DBs must open from read-only mounts
            # without creating WAL/SHM sidecars in the installation directory.
            target.commit()
            assert target.execute("pragma journal_mode=delete").fetchone() == (
                "delete",
            )
            assert target.execute("pragma integrity_check").fetchone() == ("ok",)
    native = ["gfx1013_20.HIP.fdb.txt", "gfx1013_20.db.txt", "gfx1013_20.kdb"]
    # Preserve the existing shared heuristics in the test system-DB directory.
    # Deployment copies only the three native payloads, leaving shared files alone.
    shared = {}
    for name in ["lgbm_model_meta.json", "lgbm_pcfg.bin", "lgbm_rank.bin"]:
        original = ws / "build/dist/rocm/share/miopen/db" / name
        shutil.copy2(original, db / name)
        shared[name] = sha(original)
    for p in db.iterdir():
        p.chmod(0o444)
    manifest = dict(
        status="collected_cpu_checked",
        source=source,
        device="gfx1013:xnack-",
        db_key="gfx1013_20",
        wave_size=32,
        physical_cu=40,
        hip_wgp_count=20,
        miopen_version="3.6.2",
        dtype_compute_scope="Recorded NCHW FP32/FP16/BF16 convolution fixtures only",
        finddb_keys=16,
        cpu_checked_immediate_solutions=solutions,
        perfdb_records=1,
        kernel_records=len(rows),
        files={name: sha(db / name) for name in native},
        shared_heuristics=shared,
        collection_labels=[
            "miopen-a03-collect",
            "miopen-a03-tune",
            "miopen-a03-tune-normal",
        ],
        searched_case=tune["results"][0],
        tested_cases=collection["results"] + normal["results"],
        kernel_index=[
            dict(
                name=x[0],
                args_sha256=hashlib.sha256(x[1].encode()).hexdigest(),
                hash=x[2],
                uncompressed_size=x[3],
            )
            for x in rows
        ],
        limitations="One genuinely searched native ConvHipDirectFwd performance record; not global tuning. Kernel cache includes compiled search candidates, not a claim that every blob/configuration was numerically checked. Only listed immediate API executions were CPU checked. No cross-board, compiler-version or universal performance qualification. No empty/fabricated DB or architecture renaming.",
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in [
                    "finddb_keys",
                    "cpu_checked_immediate_solutions",
                    "perfdb_records",
                    "kernel_records",
                    "files",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
