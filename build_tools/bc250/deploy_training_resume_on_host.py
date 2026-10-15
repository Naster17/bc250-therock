#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Install tested autocast handling and the reproduced LoRA resume correction."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def replace(src,dst):
    stat=dst.stat();temporary=dst.with_name('.'+dst.name+'.training-update')
    shutil.copy2(src,temporary);os.chown(temporary,stat.st_uid,stat.st_gid)
    temporary.chmod(stat.st_mode & 0o777);temporary.replace(dst)


def main():
    assert os.geteuid()==0
    ws=Path('/home/nik/bc250work/rocm-10.2-gfx1013')
    assert not subprocess.check_output(['podman','ps','-q'],text=True).strip()
    assert json.loads((ws/'logs/training-amp-tiled-p4-r2/result.json').read_text())['status']=='passed'
    result=json.loads((ws/'logs/training-app-resume-before-p4/result.json').read_text())
    assert result['runs']==[{'phase':'fresh','returncode':0},{'phase':'resume','returncode':1}]
    app=Path('/home/nik/finetune-n7/scripts/train_lora.py')
    expected=json.loads((ws/'helpers/p4-callers-before-sha256.json').read_text())['train_lora.py']
    assert sha(app)==expected,'User script changed: inspect before applying'
    module=Path('torch/nn/attention/_gfx1013.py')
    installed=json.loads((ws/'logs/attention-priorities-installed.json').read_text())
    assert sha(ws/'pytorch-work'/module)==installed[str(ws/'pytorch-work'/module)]
    assert (ws/'pytorch-source'/module).read_bytes()==(ws/'pytorch-work'/module).read_bytes()
    recovery=ws/'recovery/training-expansion-before';recovery.mkdir()
    shutil.copy2(app,recovery/'train_lora.py')
    shutil.copy2(ws/'pytorch-work'/module,recovery/'_gfx1013.py')
    for tree in ['pytorch-source','pytorch-work']:
        replace(ws/'helpers/gfx1013_attention_amp.py',ws/tree/module)
    replace(ws/'helpers/p4-train_lora.py',app)
    files=[app,ws/'pytorch-work'/module]
    (ws/'logs/training-expansion-installed.json').write_text(json.dumps({str(p):sha(p) for p in files},indent=2)+'\n')
    print('AUTOCAST_AND_LORA_RESUME_FIX_INSTALLED',flush=True)


if __name__=='__main__':main()
