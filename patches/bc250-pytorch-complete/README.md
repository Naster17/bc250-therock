# Complete BC-250 Torch source deltas (D46)

This is a separate complete series. **Do not apply it after the historical
`../bc250-pytorch` patches.** The old series is preserved as milestone evidence.

Starting baseline: `c779020750d7dc99a39ea8c104ef8df1fc9d0755`.

1. `0001-qualified-native-torch.patch` reconstructs the exact tracked tree of
   `a5249802372882dedccb91d3fd4d73be9fcdf2bd`: the native daily/RCCL source revision.
2. Optional `0002-agent-exit-lifetime.patch` reconstructs the exact tracked tree of
   `8d8b18df558cb478727ba2179f2e01dfce20ea40`, the local source head. Current D09
   launchers apply its source-derived Python agent fix through the pinned bridge;
   the native installed Torch revision remains a524980. This is not an instruction
   to replace that installed package or bypass qualification.

These are Git binary-capable diffs. In a **new private checkout** at the baseline,
use `git apply --check` and `git apply` in the listed order. Commit metadata will
differ from the original history; exact tree hashes, not new commit IDs, establish
reconstruction. `files.json` records all35 changed paths at baseline/native/head;
`revisions.json` records exact pins. Existing repositories and worktrees need not
be changed to verify either complete tree:

```sh
python3 therock/build_tools/bc250/verify_torch_d46.py --output /tmp/torch-d46.json
```

That verifier uses private shared bare clones/indexes and requires existing local
baseline Git objects. It is not a portable bundle or a full build. Upstream files,
submodules, dependencies, licenses, toolchain and build configuration must still
be supplied from their pinned sources; PLAN-07 remains open for that closure.

`generated-reference/hipify-output.diff` is **not part of this patch series**.
It retains the five observed post-HIPify source files from both installed .104
cohorts. Applying it in a scratch directory reproduces those observed bytes; it
does not independently prove code generation. Do not apply it before building:
the documented pip-only Torch build owns HIPify/code generation. Raw source and
post-codegen source are distinct audit stages.

No native rebuild, deployment, GPU behavior or performance change is claimed.
[D46 proof, source inventory and remaining limits](../../docs/development/bc250_source_d46_results/README.md).
