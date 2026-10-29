#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Record the missing installed group-major FP16 factory and isolated replacement object."""
import argparse
from pathlib import Path
import subprocess
from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS


def symbols(path):
    wanted = {
        name: []
        for name in (
            "add_device_grouped_conv2d_fwd_dl_nhwgc_gkyxc_nhwgk_f16_instances",
            "add_device_grouped_conv2d_fwd_dl_gnhwc_gkyxc_gnhwk_f16_instances",
        )
    }
    proc = subprocess.Popen(
        ["timeout", "90", "nm", "-C", "--defined-only", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    for line in proc.stdout:
        for name in wanted:
            if name + "(" in line:
                wanted[name].append(line.strip())
    assert proc.wait() == 0
    return wanted


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build-attempt", required=True)
    args = p.parse_args()
    archive = (
        WS / "build/math-libs/composable_kernel/stage/lib/libdevice_conv_operations.a"
    )
    obj = WS / f"tmp/ck-plugin-b079/build-{args.build_attempt}/gnhwc-f16-instance.o"
    a = symbols(archive)
    b = symbols(obj)
    names = list(a)
    assert len(a[names[0]]) == 1 and len(a[names[1]]) == 0
    assert len(b[names[1]]) == 1
    atomic_json(
        WS / f"tmp/ck-plugin-b079/archive-audit-{args.build_attempt}.json",
        dict(
            status="passed",
            installed_archive_sha256=sha(archive),
            isolated_object_sha256=sha(obj),
            archive_symbols=a,
            isolated_symbols=b,
            installed_archive_changed=False,
        ),
    )
    print("CK_PLUGIN_B079_ARCHIVE_AUDIT_PASS", flush=True)


if __name__ == "__main__":
    main()
