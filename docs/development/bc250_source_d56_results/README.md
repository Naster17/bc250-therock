# PLAN-07 / D56: generated-file and license provenance closure

Completed 2026-11-11. CPU-only analysis on the workstation plus read-only
.104 inspection; no build, no container, no GPU workload, no board
reservation, .105 unaccessed.
[Receipts](artifacts.json), [evidence](../../../../kernel/artifacts/source-d56).
**PLAN-07 remains OPEN** for the scope decisions below; the Torch
rebuild-path provenance is now closed.

## License closure

The D55 editable wheel collected **87 unique license files** (174 log
entries, each file logged twice; identical sets for daily and rccl-single).
All 87 exist in the D52 kit and are hashed in
[license-manifest.json](../../../../kernel/artifacts/source-d56/license-manifest.json)
(path, bytes, SHA-256, git blob OID). Preservation binding:

| Class | Count | Binding |
| --- | --- | --- |
| D47 tracked license OIDs | 44 | exact git blob match (43 dependency licenses + parent LICENSE) |
| Nested gitlink licenses | 41 | inside D50 available recursive payloads + D52 kit |
| Parent-tracked vendored dirs | 2 | llvm-openmp, miniz-3.0.2 in the D47 baseline payload + kit |

Zero OID mismatches, zero uncovered files. D47's tracked set enumerated
gitlink-top-level licenses; the 43 additional wheel licenses are nested
third-party texts (fbgemm/external/*, ideep/mkl-dnn, ...) already preserved
inside the recursive payloads. This closes the wheel license inventory that
D50 left open; it is still not a legal/SPDX review.

## Configuration and version identity

[configuration-identity.json](../../../../kernel/artifacts/source-d56/configuration-identity.json):
the D55 CMake caches were diffed against the **original** caches preserved
in D50's evidence tarball (SHAs re-verified: daily `2406c7db…`, rccl-single
`9c153e7e…`). Both profiles show only four non-path differences, all
benign: the CTest `SITE` container hostname, two `BOOL→INTERNAL` type
normalizations with identical values (`USE_C10D_XCCL=ON`, `USE_XCCL=OFF`),
and an empty find-module artifact. **Every compile-relevant flag is
identical.** Caveat recorded: the live `pytorch-work/build` cache on .104
was reconfigured after D50 (`USE_DISTRIBUTED=0`) and is not the original;
the D50-preserved copy is authoritative.

[version-metadata.json](../../../../kernel/artifacts/source-d56/version-metadata.json):
recipe `PYTORCH_BUILD_VERSION=2.13.0a0+gfx1013` → generated `torch/version.py`
(text SHA `0899cb9f…`) is identical to the retained original bound by D50
and to the live deployed daily runtime (read-only check). The original build
was already Gitless (`git_version='Unknown'`), so the Gitless kit reproduces
the metadata exactly.

## Generated-file inventory

[generated-files.json](../../../../kernel/artifacts/source-d56/generated-files.json)
enumerates the build-time generation steps from the kit build system:
hipify (`build_amd.py` via the preserved toolchain), torchgen codegen
(`cmake/Codegen.cmake` `GEN_COMMAND` with `GEN_ROCM_FLAG`, inputs
`native_functions.yaml`/`derivatives.yaml`/`torchgen/*.py`), version
metadata, caffe2 `macros.h.in`, linker version script, and the offline six
input. Every input is preserved; completeness of the closure is proven by
the D55 fully offline rebuild (no network, no undeclared mounts). Sampled
outputs are byte-verified (22/22 HIP translations, Eig object); a full
byte-level replay of every generated output is not claimed.

## Library deltas vs the deployed cohort — investigation record

[delta-investigation.json](../../../../kernel/artifacts/source-d56/delta-investigation.json).
Ruled out with evidence:

- **Flags/configuration** — identical (above).
- **Embedded source paths** — the deployed `libtorch_cpu.so` strings show
  `/workspace/pytorch-work/…`, the same path the kit recipe uses.
- **Git metadata** — both sides `git_version='Unknown'`.
- **Build timestamps** — no date strings in the deployed library; only four
  kit files use `__DATE__`/`__TIME__` (CK dispatcher bindings, civetweb ×2,
  mimalloc `options.c`), none linked into this library.

The deltas are therefore confined to objects recompiled from the ccache
**miss set** (daily 241 marker lines / 209 distinct invocations; rccl-single
251 / 224). Of these, 52 per profile are configure-time TryCompile and
hipblaslt probes that never enter the libraries, and 33 daily / 35 rccl
invocations did not parse to a source path; the parsed remainder is
dominated by `torch/csrc` TUs (122 daily / 133 rccl) including 10 generated
`autograd/generated` files, plus ittapi/nativert/kineto.
Remaining hypotheses: working-tree drift between the deployed cohort's
build time and the final retained state captured by the kit (most likely),
or per-TU non-determinism (less likely). Definitive resolution needs a
retained rebuilt library plus per-object diff, or a torchgen replay against
the retained original generated files — recorded as a follow-up, **not** a
provenance-closure blocker. Byte-equivalence of every library is explicitly
not a PLAN-07 completion criterion; D55 already matched all admitted
references.

## PLAN-07 status

Closed for the Torch rebuild path: source hashes/configuration (D52/D54/
D55/D56), dependency provenance (D50/D51), license inventory (D56),
generated-file provenance (D56), patches (D47/D52), no undeclared inputs
(D54/D55). Remaining scope decisions before PLAN-07 can be marked complete:

1. Whether a Triton rebuild demonstration is required (pin/patches/wheel
   provenance already preserved; not part of the D47–D55 chain).
2. Whether the library-delta follow-up (retained rebuild + object diff) is
   wanted before release, or accepted as a documented boundary.

Then PLAN-08 release assembly. Four important packages remain (PLAN-07..10).