#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Install tested causal optimization and Python crash traces under runtime lock."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def replace(src,dst):
    stat=dst.stat()
    temp=dst.with_name('.'+dst.name+'.attention-update')
    shutil.copy2(src,temp)
    os.chown(temp,stat.st_uid,stat.st_gid)
    temp.chmod(stat.st_mode & 0o777)
    temp.replace(dst)


def main():
    assert os.geteuid()==0
    ws=Path('/home/nik/bc250work/rocm-10.2-gfx1013')
    assert not subprocess.check_output(['podman','ps','-q'],text=True).strip()
    for name in ['attention-opt-long-p2','attention-opt-bench-p2-r2','attention-opt-api-p2']:
        result=json.loads((ws/'logs'/name/'result.json').read_text())
        assert result['status']=='passed'
        assert json.loads((ws/'logs'/name/'exit.json').read_text())=={'returncode':0,'kernel_log_unchanged':True}
    recovery=ws/'recovery/attention-priorities-before'
    recovery.mkdir()
    expected='227a68174927397cf01132421f7d5c7dd7c2a65ba05bbc4babc651addde2bbc3'
    module=Path('torch/nn/attention/_gfx1013.py')
    for tree in ['pytorch-source','pytorch-work']:
        assert sha(ws/tree/module)==expected
    shutil.copy2(ws/'pytorch-work'/module,recovery/'_gfx1013.py')
    launcher=Path('/usr/local/bin/bc250-torch')
    helper=ws/'helpers/bc250_torch.py'
    assert launcher.read_bytes()==helper.read_bytes()
    shutil.copy2(launcher,recovery/'bc250-torch')
    shutil.copy2(helper,recovery/'bc250_torch.py')
    for tree in ['pytorch-source','pytorch-work']:
        replace(ws/'helpers/gfx1013_attention_optimized.py',ws/tree/module)
    candidate=ws/'helpers/bc250_torch.py.candidate'
    replace(candidate,launcher)
    replace(candidate,helper)
    files=[ws/'pytorch-work'/module,launcher,helper]
    (ws/'logs/attention-priorities-installed.json').write_text(json.dumps({str(p):sha(p) for p in files},indent=2)+'\n')
    print('CAUSAL_OPTIMIZATION_AND_FAULT_TRACES_INSTALLED',flush=True)


if __name__=='__main__':main()
