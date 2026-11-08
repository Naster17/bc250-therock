#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Actual positive registration and corruption/refusal checks on inactive copies."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = a.cohort_root
    sys.path.insert(0, str(root / "tools"))
    import bc250_ck as ck
    import bc250_migraphx as mx
    import bc250_ort as ort
    from bc250_torch import LaunchOptions

    results = []
    for profile in ["daily", "rccl-single"]:
        opts = LaunchOptions(runtime_profile=profile)
        for module in [ck, mx, ort]:
            module.registered(opts)
        results.append("all-three-actual-registrations-" + profile)
    fixtures = root / "identity-refusals"
    fixtures.mkdir()

    def refuse(label, fn, expected):
        try:
            fn()
        except ValueError as exc:
            assert expected in str(exc), (label, str(exc))
            results.append(label)
        else:
            raise AssertionError("invalid identity accepted: " + label)

    cr = json.loads(ck.REGISTRY.read_text())
    copied = fixtures / "copied-ck"
    shutil.copytree(cr["package"], copied)
    m = json.loads((copied / "manifest.json").read_text())
    relative = next(k for k in m["files"] if k.endswith(".so"))
    f = copied / relative
    before = f.read_bytes()
    changed = bytearray(before)
    changed[64] ^= 1
    f.write_bytes(changed)
    refuse(
        "changed-private-plugin-refused",
        lambda: ck.check_package(copied, cr),
        "CK payload identity mismatch",
    )
    f.write_bytes(before)
    (copied / "undeclared-file").write_text("controlled")
    refuse(
        "extra-package-file-refused",
        lambda: ck.check_package(copied, cr),
        "CK package file set mismatch",
    )
    ore = json.loads(ort.REGISTRY.read_text())
    wrong = copy.deepcopy(ore)
    wrong["index_sha256"] = "0" * 64
    refuse(
        "ORT-index-refused",
        lambda: ort.check_package(Path(ore["package"]), wrong),
        "ORT index/manifest identity mismatch",
    )
    original = mx.REGISTRY
    wrong = json.loads(original.read_text())
    wrong["runtime_sha256"]["daily"] = "0" * 64
    tmp = fixtures / "wrong-sdk-runtime.json"
    tmp.write_text(json.dumps(wrong))
    mx.REGISTRY = tmp
    refuse(
        "SDK-runtime-refused",
        lambda: mx.registered(LaunchOptions()),
        "SDK runtime identity mismatch",
    )
    mx.REGISTRY = original
    original = ort.REGISTRY
    wrong = copy.deepcopy(ore)
    wrong["sdk_registration"]["runtime_sha256"]["daily"] = "0" * 64
    tmp = fixtures / "wrong-ORT-sdk.json"
    tmp.write_text(json.dumps(wrong))
    ort.REGISTRY = tmp
    refuse(
        "ORT-embedded-SDK-registration-refused",
        lambda: ort.registered(LaunchOptions()),
        "ORT SDK/runtime identity mismatch",
    )
    ort.REGISTRY = original
    # Every original native/provider identity remains unmodified by refusals.
    for module in [ck, mx, ort]:
        module.registered(LaunchOptions())
    stage = json.loads((root / "stage.json").read_text())
    for group in ["registries", "tools"]:
        for path, sha in stage[group].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha, path
    results.append("original-registrations-and-generated-tool-bytes-preserved")
    a.output.write_text(
        json.dumps(dict(status="passed", checks=results), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
