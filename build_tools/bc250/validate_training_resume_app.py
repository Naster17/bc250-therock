#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exercise the actual user LoRA CLI on a disposable tiny model and English fixture."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from transformers.utils import logging


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--expect-fixed',action='store_true')
    ap.add_argument('--legacy-only',action='store_true')
    args=ap.parse_args()
    logging.disable_progress_bar()
    root=args.output.parent
    if args.legacy_only:
        old=Path('/home/nik/bc250work/rocm-10.2-gfx1013/logs/training-app-resume-before-p4')
        checkpoint=old/'fresh/step-1'
        state=torch.load(checkpoint/'training-state.pt',map_location='cpu',weights_only=True)
        assert 'scheduler' not in state and 'python_rng' not in state
        command=[sys.executable,'/home/nik/finetune-n7/scripts/train_lora.py','--data',str(old/'data'),
            '--model',str(old/'model'),'--resume',str(checkpoint),'--out',str(root/'resume'),
            '--max-seq','32','--max-steps','1','--accum','1','--val-subset','1','--eval-every','1',
            '--log-every','1','--rank','2','--alpha','4','--warmup','1','--schedule','const',
            '--attention-backend','tiled']
        with (root/'legacy.log').open('w') as log:
            code=subprocess.call(command,stdout=log,stderr=subprocess.STDOUT,timeout=90)
        assert code==0
        log=(root/'legacy.log').read_text()
        assert 'Legacy checkpoint lacks scheduler state' in log and 'Legacy checkpoint lacks Python/data-order state' in log
        from safetensors.torch import load_file
        before=load_file(str(checkpoint/'adapter_model.safetensors'))
        after=load_file(str(root/'resume/step-2/adapter_model.safetensors'))
        assert any(not torch.equal(before[n],after[n]) for n in before)
        args.output.write_text(json.dumps({'status':'passed','legacy_resume_returncode':code,
            'adapter_updated':True,'exact_legacy_replay_claimed':False},indent=2)+'\n')
        print('LEGACY_LORA_CHECKPOINT_RESUME_PASS',flush=True)
        return
    modeldir=root/'model';modeldir.mkdir()
    tok=AutoTokenizer.from_pretrained('/home/nik/bc250work/rocm-10.2-gfx1013/training-model',local_files_only=True)
    tok.save_pretrained(modeldir)
    config=Qwen3Config(vocab_size=len(tok),hidden_size=64,intermediate_size=96,num_hidden_layers=1,
        num_attention_heads=4,num_key_value_heads=2,head_dim=16,attention_dropout=0.,eos_token_id=tok.eos_token_id)
    torch.manual_seed(719)
    Qwen3ForCausalLM(config).save_pretrained(modeldir)
    data=root/'data';data.mkdir()
    fixture={'turns':[{'role':'toma','text':'Describe this test.'},
                      {'role':'n7','text':'The GPU trains attention and saves a checkpoint.'}]}
    for name in ['train.jsonl','val.jsonl']:(data/name).write_text(json.dumps(fixture)+'\n')
    script='/home/nik/finetune-n7/scripts/train_lora.py'
    command=[sys.executable,script,'--data',str(data),'--model',str(modeldir),
        '--max-seq','32','--max-steps','1','--accum','1','--val-subset','1','--eval-every','1',
        '--log-every','1','--rank','2','--alpha','4','--warmup','1','--schedule','const',
        '--attention-backend','tiled']
    report={'status':'running','runs':[]}
    for phase,extra in [('fresh',[]),('resume',['--resume',str(root/'fresh/step-1')])]:
        with (root/(phase+'.log')).open('w') as log:
            code=subprocess.call(command+['--out',str(root/phase)]+extra,
                stdout=log,stderr=subprocess.STDOUT,timeout=90)
        report['runs'].append({'phase':phase,'returncode':code})
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        if phase=='fresh':assert code==0
    if args.expect_fixed:
        assert report['runs'][-1]['returncode']==0
        state=torch.load(root/'resume/step-2/training-state.pt',map_location='cpu',weights_only=True)
        assert state['opt_step']==2 and state['step']==2
        assert 'scheduler' in state and state['gpu_rng'].device.type=='cpu'
        assert 'python_rng' in state and 'remaining_indices' in state
        assert 'trainable params: 0' not in (root/'resume.log').read_text()
        report['status']='passed'
    else:
        assert report['runs'][-1]['returncode']!=0, 'Expected defect did not reproduce; inspect before editing'
        report['status']='resume defect reproduced; original fixture/log preserved'
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('TRAINING_APP_RESUME',report,flush=True)


if __name__=='__main__':main()
