#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Use the existing hardware TF32 capability API instead of CUDA major numbers."""
import os
from pathlib import Path
import shutil


def main():
    assert os.geteuid()==0
    ws=Path('/home/nik/bc250work/rocm-10.2-gfx1013')
    relative=Path('torch/_inductor/compile_fx.py')
    candidate=ws/'helpers/compile_fx.py.candidate'
    recovery=ws/'recovery/tf32-diagnostic-before';recovery.mkdir()
    for tree in ['pytorch-source','pytorch-work']:
        p=ws/tree/relative
        old=p.read_text()
        assert old.replace('and torch.cuda.get_device_capability() >= (8, 0)',
                           'and torch.cuda.is_tf32_supported()')==candidate.read_text()
        shutil.copy2(p,recovery/(tree+'-compile_fx.py'))
        temp=p.with_suffix('.sdpa-new');shutil.copy2(candidate,temp);temp.replace(p)
    print('TF32_CAPABILITY_DIAGNOSTIC_FIXED',flush=True)


if __name__=='__main__':main()
