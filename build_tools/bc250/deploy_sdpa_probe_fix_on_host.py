#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Stage the reviewed probe diagnostic fix; call under the exclusive runtime lock."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install-built', action='store_true')
    args = parser.parse_args()
    ws = Path('/home/nik/bc250work/rocm-10.2-gfx1013')
    assert os.geteuid() == 0
    assert not subprocess.check_output(['podman', 'ps', '-q'], text=True).strip()
    torch = ws/'pytorch-work'
    source = torch/'aten/src/ATen/native/transformers/cuda/sdp_utils.cpp'
    candidate = ws/'helpers/sdp_utils.cpp.candidate'
    old, new = source.read_text(), candidate.read_text()
    needle = '  TORCH_WARN_ONCE(!debug, "Torch was not compiled with memory efficient attention.");'
    replacement = '  if (debug) {\n    TORCH_WARN("Torch was not compiled with memory efficient attention.");\n  }'
    recovery = ws/'recovery/sdpa-probe-before'
    lib = torch/'torch/lib/libtorch_hip.so'
    if args.install_built:
        assert old == new and recovery.is_dir()
        assert sha(lib) == json.loads((recovery/'identity.json').read_text())[str(lib)]
    else:
        assert old.count(needle) == 1 and old.replace(needle, replacement) == new
        recovery.mkdir()
        for p in [source, lib]:
            shutil.copy2(p, recovery/p.name)
        (recovery/'identity.json').write_text(json.dumps({str(p):sha(p) for p in [source, lib]}, indent=2)+'\n')
        shutil.copyfile(candidate, source)
    manifest = json.loads(Path('/opt/bc250-stack/current/runtime.json').read_text())
    command = ['podman', 'run', '--rm', '--name', 'bc250-sdpa-probe-build',
               '--network', 'none', '-v', f'{ws}:/workspace:rw',
               manifest['image'], 'bash', '/workspace/helpers/build_torch_sdpa.sh']
    code = 0
    if not args.install_built:
        with (ws/'logs/sdpa-torch-build.log').open('w') as log:
            code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
        (ws/'logs/sdpa-torch-build.exit').write_text(str(code)+'\n')
    if code:
        # Keep generated source/build failure evidence and restore the working
        # source+library. No user workload can enter until this lock is released.
        shutil.copy2(recovery/source.name, source)
        shutil.copy2(recovery/lib.name, lib)
        raise SystemExit(code)
    # The Ninja target writes build/lib; editable Torch imports torch/lib.
    built = torch/'build/lib/libtorch_hip.so'
    temporary = lib.with_name('.libtorch_hip.sdpa-install.so')
    shutil.copy2(built, temporary)
    temporary.replace(lib)
    assert sha(built) == sha(lib)
    (ws/'logs/sdpa-probe-build-identity.json').write_text(json.dumps(
        {str(p):sha(p) for p in [source,lib]}, indent=2)+'\n')


if __name__ == '__main__':
    main()
