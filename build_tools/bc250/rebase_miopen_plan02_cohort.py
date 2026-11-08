#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Finalize new, inactive package identities after generating their launcher sources."""
import hashlib
import json
from pathlib import Path
import tarfile


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(p, x):
    p.write_text(json.dumps(x, indent=2) + "\n")


def finalize(root):
    tools = root / "tools"
    regs = root / "registries"
    mx = json.loads((regs / "migraphx.json").read_text())
    for n in mx["launcher_sha256"]:
        mx["launcher_sha256"][n] = sha(tools / n)
    write(regs / "migraphx.json", mx)
    ck = json.loads((regs / "miopen-dl.json").read_text())
    for n in ck["launcher_sha256"]:
        ck["launcher_sha256"][n] = sha(tools / n)
    write(regs / "miopen-dl.json", ck)
    ore = json.loads((regs / "ort-migraphx.json").read_text())
    pkg = Path(ore["package"])
    mp = pkg / "share/bc250-ort/manifest.json"
    ip = pkg / "share/bc250-ort/sha256.json"
    om = json.loads(mp.read_text())
    om["sdk_registration"] = mx
    write(mp, om)
    oi = json.loads(ip.read_text())
    oi["files"]["share/bc250-ort/manifest.json"] = sha(mp)
    write(ip, oi)
    archive = root / "ort-plan02-final.tar.gz"
    assert not archive.exists()
    with tarfile.open(archive, "x:gz") as t:
        t.add(pkg, arcname=pkg.name)
    ore.update(
        sdk_registration=mx,
        index_sha256=sha(ip),
        manifest_sha256=sha(mp),
        archive_sha256=sha(archive),
    )
    for n in ore["launcher_sha256"]:
        ore["launcher_sha256"][n] = sha(tools / n)
    write(regs / "ort-migraphx.json", ore)
