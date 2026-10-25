from pathlib import Path
import json, hashlib, tarfile, shutil, os

ws = Path("/home/nik/bc250work/rocm-10.2-gfx1013")
base = ws / "tmp/migraphx-b067-native"
incoming = base / "incoming"
fs = os.statvfs(base)
available = fs.f_bavail * fs.f_frsize
assert available > 4 * 1024**3
meta = json.loads((incoming / "upstream-package.json").read_text())
archive = incoming / "sdk.tar.gz"
assert hashlib.sha256(archive.read_bytes()).hexdigest() == meta["archive_sha256"]
package = base / "package-r3"
package.mkdir()
roots = []
for folder in ("relocated-A", "relocated B with spaces"):
    dest = package / folder
    dest.mkdir()
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")
    sdk = dest / "bc250-migraphx-2.18.0.dev-gfx1013-cp312-b067"
    checks = json.loads((sdk / "share/bc250-sdk/sha256.json").read_text())
    for name, digest in checks["files"].items():
        assert hashlib.sha256((sdk / name).read_bytes()).hexdigest() == digest, name
    for name, target in checks["symlinks"].items():
        assert os.readlink(sdk / name) == target, name
    roots.append(str(sdk))
meta.update(
    roots=roots,
    archive=str(archive),
    original_external_roots=meta["roots"],
    native_stage=True,
)
(package / "final.json").write_text(json.dumps(meta, indent=2) + "\n")
consumer = base / "consumer-r2/build"
consumer.mkdir(parents=True)
shutil.copyfile(incoming / "consumer", consumer / "consumer")
(consumer / "consumer").chmod(0o755)
shutil.copyfile(incoming / "source-r5.json", base / "source-r5.json")
fs = os.statvfs(base)
remaining = fs.f_bavail * fs.f_frsize
assert remaining > 2 * 1024**3 and available - remaining < 2 * 1024**3
receipt = dict(
    status="passed",
    archive_sha256=meta["archive_sha256"],
    roots=roots,
    ordinary_bytes_before=available,
    ordinary_bytes_after=remaining,
    growth_limit_bytes=2 * 1024**3,
    ordinary_floor_bytes=2 * 1024**3,
    original_build_mount=False,
)
(base / "stage.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
