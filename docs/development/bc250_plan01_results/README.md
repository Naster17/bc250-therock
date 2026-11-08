# PLAN-01: selected upcoming GPU gate reservation

Completed 2026-11-08. Implementation2e24474d6. Shared board_preflight.py replaces
historical training preflight imports in B07.11 native build/qualification and
B07.12 package/qualification. The shared qualify_rccl_b038 gate now takes a
cooperative EX board reservation and SH runtime lock, checks default-local Docker,
Podman, GPU fds, memory>8GiB and active watchdog, and records post-test Docker
state/errors. Unknown or busy Docker state refuses qualification. No foreign stop.

Both hosts contain additive tools/plan01 overlays; older helper snapshots remain
untouched. PLAN-02 must select this overlay BEFORE legacy helpers in PYTHONPATH,
not execute old copied guardians directly. .104 legacy helper root is
/home/nik/bc250work/rocm-10.2-gfx1013/helpers. The overlay includes the selected
updated gate/build/qualification modules and both shared preflight modules.

Four real CPU-only controls per host pass: idle admission, competing reservation
refusal, prefix EX writer refusal while SH-held, exception release/reacquisition.
[Results](results.json). 62 local tests include four busy/unknown/nonroot controls;
Black, critical Ruff+F401 and diff pass. No numerical GPU tests or new recovery
reboot performed. Existing gate600s recovery/540s child/15s termination/ACK-after-
container-exit code retained; D25 recovery evidence remains qualified separately.
The old gate resource limits and selected launcher remain unchanged.

Kernel logs/registry/boots unchanged, boards idle. No runtime install, native
rebuild, application access or foreign process stop. Reservation protects only
cooperating callers. Custom/rootless Docker daemons and future uncooperating
workloads are not exhaustively excluded; old unused guardians were not migrated.
Next PLAN-02: read-only current loaded MIOpen vs B07.11/B07.12 artifact audit,
then bounded original-failure/repair and compatibility/rollback qualification.
