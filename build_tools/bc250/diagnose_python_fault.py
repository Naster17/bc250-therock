#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Instrument the historical mixed-operator/sparse crash area in fresh processes."""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--scope', choices=['mixed','sparse'], default='mixed')
    args = ap.parse_args()
    faulthandler.enable(all_threads=True)
    directory = args.output.parent
    report = dict(status='running', attempts=[], original_signature='IP=0x3b3e6; no core/backtrace retained',
                  core_pattern=Path('/proc/sys/kernel/core_pattern').read_text().strip(),
                  core_limit=resource.getrlimit(resource.RLIMIT_CORE))
    operator = Path(__file__).with_name('validate_operator_coverage.py')
    assert operator.exists()
    environment = {**os.environ, 'PYTHONFAULTHANDLER':'1', 'PYTHONMALLOC':'debug',
                   'MALLOC_CHECK_':'3', 'MALLOC_PERTURB_':'165'}
    for iteration in range(8):
        case = directory/f'process-{iteration}'
        case.mkdir()
        started = time.monotonic()
        with (case/'operator.log').open('w') as output:
            command = ([sys.executable, '-X', 'faulthandler', '-X', 'dev', str(operator),
                        '--output', str(case/'operator.json')] if args.scope=='mixed' else
                       [sys.executable, '-X', 'faulthandler', '-X', 'dev',
                        str(Path(__file__)), '--sparse-child', str(case)])
            code = subprocess.call(command, env=environment,
                cwd=case, stdout=output, stderr=subprocess.STDOUT, timeout=50)
        entry = dict(iteration=iteration, returncode=code, seconds=time.monotonic()-started,
                     core_files=[str(p) for p in case.glob('core*')])
        if code == 0:
            result = json.loads((case/'operator.json').read_text())
            assert result['status']=='passed'
            entry['reference_checks']=len(result['checks'])
            assert entry['reference_checks']==(55 if args.scope=='mixed' else 240)
        report['attempts'].append(entry)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print('PYTHON_FAULT_PROBE', entry, flush=True)
        assert code == 0, 'Preserve logs/core; stop and analyze failure before continuing'
    report['status']='not reproduced in bounded diagnostic run; historical cause unresolved'
    args.output.write_text(json.dumps(report,indent=2)+'\n')


def sparse_child(directory):
    import torch
    faulthandler.enable(all_threads=True)
    torch.set_num_threads(4)
    torch.manual_seed(718)
    report={'status':'running','checks':[]}
    for iteration in range(20):
        for dtype in [torch.float32,torch.float64]:
            a=torch.randn(13,7,dtype=dtype)
            tol=4e-4 if dtype==torch.float32 else 1e-10
            for name,fn,arg in [('svd',torch.linalg.svdvals,a),
                                ('eigen',torch.linalg.eigvalsh,a.T@a+torch.eye(7)*7),
                                ('fft',torch.fft.fft,a[:,0])]:
                (directory/'stage.json').write_text(json.dumps({'iteration':iteration,'dtype':str(dtype),'operation':name}))
                torch.testing.assert_close(fn(arg.cuda()).cpu(),fn(arg),rtol=tol,atol=tol)
                report['checks'].append(name)
            dense=torch.randn(11,13,dtype=dtype)
            dense[dense.abs()<1]=0
            right=torch.randn(13,5,dtype=dtype)
            for layout in [torch.sparse_csr,torch.sparse_coo]:
                (directory/'stage.json').write_text(json.dumps({'iteration':iteration,'dtype':str(dtype),'operation':str(layout)}))
                sparse=dense.to_sparse(layout=layout)
                torch.testing.assert_close(torch.sparse.mm(sparse.cuda(),right.cuda()).cpu(),dense@right,rtol=tol,atol=tol)
                report['checks'].append(str(layout))
            torch.testing.assert_close((a.cuda()@a.cuda().T).cpu(),a@a.T,rtol=tol,atol=tol)
            report['checks'].append('gemm')
        (directory/'operator.json').write_text(json.dumps(report))
    report['status']='passed'
    (directory/'operator.json').write_text(json.dumps(report))


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--sparse-child':sparse_child(Path(sys.argv[2]))
    else:main()
