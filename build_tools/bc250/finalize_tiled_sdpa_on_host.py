#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Record tested component identities under the exclusive prefix-update lock."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--torch-revision', required=True)
    ap.add_argument('--therock-revision', required=True)
    args = ap.parse_args()
    assert os.geteuid() == 0
    ws = Path('/home/nik/bc250work/rocm-10.2-gfx1013')
    release = Path('/opt/bc250-stack/current').resolve()
    cases = json.loads((ws/'logs/sdpa-tiled-candidate-r5/result.json').read_text())
    integration = json.loads((ws/'logs/sdpa-tiled-installed-integration-r4/result.json').read_text())
    assert cases['status'] == integration['status'] == 'passed'
    for name in ['sdpa-tiled-candidate-r5','sdpa-tiled-installed-integration-r4','sdpa-torch-regression']:
        assert json.loads((ws/'logs'/name/'exit.json').read_text()) == {
            'returncode': 0, 'kernel_log_unchanged': True}
    recorded = json.loads((ws/'logs/sdpa-deployed-source-identity.json').read_text())
    assert all(sha(Path(name)) == digest for name,digest in recorded.items())
    old_sources = ws/'recovery/sdpa-probe-before'
    source = ws/'pytorch-source'
    work = ws/'pytorch-work'
    relative = Path('aten/src/ATen/native/transformers/cuda/sdp_utils.cpp')
    assert (source/relative).read_bytes() == (old_sources/'sdp_utils.cpp').read_bytes()
    recovery = ws/'recovery/sdpa-metadata-before'
    recovery.mkdir()
    shutil.copy2(release/'runtime.json', recovery/'runtime.json')
    for tree in [source, work]:
        p = tree/'GIT_REVISION'
        if p.exists(): shutil.copy2(p, recovery/(tree.name+'-GIT_REVISION'))
    # Keep the retained source import in sync with the two-file component commit.
    shutil.copy2(source/relative, recovery/'source-sdp_utils.cpp')
    shutil.copy2(work/relative, source/relative)
    mod = Path('torch/nn/attention/_gfx1013.py')
    assert not (source/mod).exists()
    shutil.copy2(work/mod, source/mod)
    for tree in [source, work]: (tree/'GIT_REVISION').write_text(args.torch_revision+'\n')
    runtime = json.loads((release/'runtime.json').read_text())
    runtime['previous_sdpa_runtime_revisions'] = {k:runtime[k] for k in ['therock_revision','torch_revision']}
    runtime['torch_revision'] = args.torch_revision
    runtime['therock_revision'] = args.therock_revision
    record = dict(status='bounded opt-in qualified', torch_revision=args.torch_revision,
        therock_revision=args.therock_revision, files=recorded,
        library_sha256=sha(work/'torch/lib/libtorch_hip.so'),
        source_sha256=sha(work/relative), built_in_efficient_attention=False,
        default_backend='math', cases='sdpa-tiled-candidate-r5',
        integration='sdpa-tiled-installed-integration-r4', regression='sdpa-torch-regression',
        limitations='Eager first-order finite FP32/FP16/BF16, D<=128, sequence<=4096; no dropout, mask gradients, compiled/graph qualification or performance claim; dense masks retain input storage; full Qwen3-1.7B training unqualified')
    runtime['sdpa_tiled'] = record
    runtime['source_state'] += '; Torch opt-in scalar tiled SDPA and diagnostic probe correction; libraries unchanged'
    (ws/'SOURCE_SDPA.json').write_text(json.dumps(record, indent=2)+'\n')
    shutil.copy2(ws/'SOURCE_SDPA.json', release/'SOURCE_SDPA.json')
    temporary = release/'.runtime.sdpa.json'
    temporary.write_text(json.dumps(runtime, indent=2)+'\n')
    temporary.replace(release/'runtime.json')
    print('SDPA_METADATA_RECORDED', args.torch_revision, flush=True)


if __name__ == '__main__':
    main()
