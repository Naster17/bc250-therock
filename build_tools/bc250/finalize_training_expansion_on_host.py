#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Record the bounded training/attention results and retain unresolved failures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--torch-revision',required=True)
    ap.add_argument('--therock-revision',required=True)
    args=ap.parse_args();assert os.geteuid()==0
    ws=Path('/home/nik/bc250work/rocm-10.2-gfx1013');release=Path('/opt/bc250-stack/current').resolve()
    passed=['attention-long-p1','attention-qwen-p1','attention-opt-long-p2','attention-opt-api-p2',
        'attention-opt-bench-p2-r2','attention-qwen-optimized-p4','training-app-resume-fixed-p4',
        'training-legacy-resume-p4','training-amp-resume-p4-final','training-compile-p4-r2']
    for name in passed+['python-fault-sparse-p3']:
        assert json.loads((ws/'logs'/name/'exit.json').read_text())=={'returncode':0,'kernel_log_unchanged':True}
    assert json.loads((ws/'logs/python-fault-diagnostics-p3/exit.json').read_text())['returncode']==1
    relative=[Path('torch/nn/attention/_gfx1013.py'),Path('torch/_inductor/compile_fx.py')]
    for path in relative:assert (ws/'pytorch-source'/path).read_bytes()==(ws/'pytorch-work'/path).read_bytes()
    recovery=ws/'recovery/training-expansion-metadata-before';recovery.mkdir()
    for name in ['runtime.json','SOURCE_SDPA.json']:shutil.copy2(release/name,recovery/name)
    for tree in ['pytorch-source','pytorch-work']:
        p=ws/tree/'GIT_REVISION';shutil.copy2(p,recovery/(tree+'-GIT_REVISION'))
        p.write_text(args.torch_revision+'\n')
    runtime=json.loads((release/'runtime.json').read_text())
    record=dict(torch_revision=args.torch_revision,therock_revision=args.therock_revision,
        status={'priority1':'bounded long/reference and real Qwen LoRA checks passed',
                'priority2':'causal kernel optimization bounded-qualified',
                'priority3':'investigated/instrumented; historical CPU crash unresolved',
                'priority4':'bounded AMP, compilation and checkpoint resume checks passed'},
        gates=passed,diagnostic='python-fault-sparse-p3; eight fresh processes/1920 checks, no reproduced crash',
        unresolved=['Historical low-address Python IP0x3b3e6 fault has no retained core/backtrace; not fixed',
            'Instrumented mixed operator run stopped before sparse operations on MIOpen invalid elapsed time / missing workspace; retained, not fixed'],
        default_backend='math',files={str(ws/'pytorch-work'/p):sha(ws/'pytorch-work'/p) for p in relative},
        application_sha256=sha(Path('/home/nik/finetune-n7/scripts/train_lora.py')),
        launcher_sha256=sha(Path('/usr/local/bin/bc250-torch')),
        library_sha256=sha(ws/'pytorch-work/torch/lib/libtorch_hip.so'),
        limitations='Synthetic-token Qwen1.7B LoRA refs at512/1024, full-model AMP/resume on small Qwen only; compiled tests use math attention, tiled compile/graphs/dropout remain unqualified; no broad speedup/convergence/stability claim')
    runtime['previous_training_expansion_runtime_revisions']={k:runtime[k] for k in ['torch_revision','therock_revision','source_manifest']}
    runtime['torch_revision']=args.torch_revision;runtime['therock_revision']=args.therock_revision
    runtime['training_expansion']=record
    runtime['sdpa_tiled'].update(torch_revision=args.torch_revision,therock_revision=args.therock_revision,
        status='bounded opt-in long/GQA/autocast qualified',files=record['files'],limitations=record['limitations'])
    runtime['source_state']+='; causal tile pruning, autocast, native TF32 diagnostics and LoRA resume state corrections'
    runtime['source_manifest']=str(release/'SOURCE_TRAINING_EXPANSION.json')
    (ws/'SOURCE_TRAINING_EXPANSION.json').write_text(json.dumps(record,indent=2)+'\n')
    shutil.copy2(ws/'SOURCE_TRAINING_EXPANSION.json',release/'SOURCE_TRAINING_EXPANSION.json')
    (release/'SOURCE_SDPA.json').write_text(json.dumps(runtime['sdpa_tiled'],indent=2)+'\n')
    temporary=release/'.runtime.training-expansion.json'
    temporary.write_text(json.dumps(runtime,indent=2)+'\n');temporary.replace(release/'runtime.json')
    print('TRAINING_EXPANSION_IDENTITIES_RECORDED',flush=True)


if __name__=='__main__':main()
