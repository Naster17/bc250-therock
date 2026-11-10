# D46 visibility proposal (historical source-only receipt)

D48 now builds and qualifies the identical five-line change as rocm-systems
`143f1d878296176c738274202092bb247b7b771c`. The active SDK is still unchanged;
[qualification and remaining limits](../../../docs/development/bc250_sdk_d48_results/README.md).
The paragraphs below retain the original D46 source-only scope.

The candidate applies to `rocm-systems` native revision
`1f9609fd69c4923291f58934495520fe62e66ef8`. The four relevant retained SDK files
match this revision; the installed DSO is exactly its retained CI build after
the recorded RPATH transformation. Source application was checked in a private
bare index. No component working tree, build, native library or prefix was edited.

The proposal treats the exact `-1` visibility token as a terminating selector
before the SDK reaches its fatal-in-CI invalid-token diagnostic. Already matched
agents retain selection; later agents terminate. CI checks are otherwise unchanged.
This is a narrow proposal, not a claim that all visibility grammar is repaired.
Empty filters, malformed/oversized tokens and UUID semantics need separate review.

Before activation: establish compiler/link/cache/scratch/staging/rollback resources,
build an isolated candidate with ccache and bounded jobs, and test direct imports
with HIP/CUDA/ROCR `-1`, valid0 and preserved ordinal/UUID termination controls.
Verify actual device-count semantics, native traces and ordinary GPU output/
backward regressions in the two cohorts. Use owned fixtures/reservation/recovery
and stop only owned work. Keep D44's production refusal during qualification.
No global CI-disable workaround or in-place build/prefix modification is authorized
by this source-only receipt.

[D46 evidence and remaining reconstruction work](../../../docs/development/bc250_source_d46_results/README.md).
