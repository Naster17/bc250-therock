#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Install the bounded-tested opt-in module and reviewed caller patch under lock."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def replace(source, destination):
    temporary = destination.with_name('.'+destination.name+'.sdpa-new')
    shutil.copy2(source, temporary)
    if destination.exists():
        s = destination.stat()
        os.chown(temporary, s.st_uid, s.st_gid)
        temporary.chmod(s.st_mode & 0o777)
    temporary.replace(destination)


def main():
    ws = Path('/home/nik/bc250work/rocm-10.2-gfx1013')
    assert os.geteuid() == 0
    assert not subprocess.check_output(['podman', 'ps', '-q'], text=True).strip()
    result = json.loads((ws/'logs/sdpa-tiled-candidate-r5/result.json').read_text())
    assert result['status'] == 'passed' and len(result['cases']) == 18
    assert json.loads((ws/'logs/sdpa-tiled-candidate-r5/exit.json').read_text()) == {
        'returncode': 0, 'kernel_log_unchanged': True}
    callers = Path('/home/nik/finetune-n7/scripts')
    expected = json.loads((ws/'helpers/sdpa-callers-before-sha256.json').read_text())
    for name, digest in expected.items():
        assert sha(callers/name) == digest, f'Caller changed: {name}; inspect before applying'
    module = ws/'pytorch-work/torch/nn/attention/_gfx1013.py'
    assert not module.exists()
    recovery = ws/'recovery/sdpa-callers-before'
    recovery.mkdir()
    for name in expected:
        shutil.copy2(callers/name, recovery/name)
    (recovery/'identity.json').write_text(json.dumps(expected, indent=2)+'\n')
    replace(ws/'helpers/gfx1013_attention_candidate.py', module)
    for name in expected:
        replace(ws/'helpers'/('sdpa-'+name), callers/name)
    record = {str(p): sha(p) for p in [module, *(callers/name for name in expected)]}
    (ws/'logs/sdpa-deployed-source-identity.json').write_text(json.dumps(record, indent=2)+'\n')
    print('TILED_SDPA_OPT_IN_DEPLOYED', flush=True)


if __name__ == '__main__':
    main()
