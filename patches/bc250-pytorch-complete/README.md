# BC-250 PyTorch — complete reconstruction series

Complete single-base patch series for the published `bc250-torch` fork
(`Naster17/bc250-torch`, branch `bc250/gfx1013`). Regenerated from the
authoritative patched source history with `git format-patch base..head`; verified
by the umbrella fork CI (`tools/verify_patch_series.py`, component `pytorch`).

- **Upstream:** `ROCm/pytorch` (the ROCm fork, *not* mainline `pytorch/pytorch`)
- **Base:** `c779020750d7dc99a39ea8c104ef8df1fc9d0755` (release/2.13)
- **Patches:** 23, ordered `0001`–`0023`
- **Reconstructed head:** `262accc8ac53c34cd649d4df49c0cf22999eb10d`
- **Reconstructed tree:** `ce52766844bf301974b9b4887072660c62ecaa12`

Patches `0001`–`0022` reconstruct head `8d8b18df558cb478727ba2179f2e01dfce20ea40`
(tree `33a862732a52c8825a3d1f9c0b609d06fba0df65`), exactly matching the verified
complete-export record. Patch `0023` adds the opt-in fused gated-delta solve/scan
(`use_fused_solve_scan`, default path unchanged).

## Reconstruct

```bash
git clone https://github.com/ROCm/pytorch.git && cd pytorch
git checkout c779020750d7dc99a39ea8c104ef8df1fc9d0755
git am /path/to/bc250-stack/patches/bc250-pytorch-complete/*.patch
git rev-parse HEAD^{tree}   # must equal ce52766844bf301974b9b4887072660c62ecaa12
```

`git am` can create different commit hashes through committer metadata; the tree
hash is the identity that matters. Building the reconstructed source follows the
offline recipes in [`../../recipes/`](../../recipes/) and
[`../../docs/BUILDING.md`](../../docs/BUILDING.md) (native gfx1013, both profiles).

## Relation to the historical series

[`../bc250-pytorch`](../bc250-pytorch/README.md) is the retained historical
*incremental* export (10 patches on shifting intermediate bases). It is incomplete
by design and **must not be combined with this series**. This directory supersedes
it for reconstruction; the old directory remains unchanged as historical evidence.
