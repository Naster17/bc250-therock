# BC-250 PyTorch patches (historical incremental export)

This is a retained **historical incremental** patch export (patches on shifting
intermediate bases). It is **incomplete by design** and is superseded by the
complete single-base reconstruction series at
[`../bc250-pytorch-complete`](../bc250-pytorch-complete/README.md).

**Do not combine these patches with the complete series.** For reconstruction,
use the complete series. This directory is kept only as a historical record of
the incremental development; the authoritative reconstructable source is the
complete series and the published `bc250-torch` fork (`bc250/gfx1013` branch).

The patches apply with `git am` to their respective intermediate bases; `git am`
can create different commit hashes through committer metadata, so the tree hash
is the identity that matters. Building reconstructed source follows the offline
recipes in [`../../recipes/`](../../recipes/) and
[`../../docs/BUILDING.md`](../../docs/BUILDING.md).