# PLAN-02: precision repair candidate qualification

2026-11-08 batch, implementation19de82f88. **PLAN-02 remains in progress; no
ordinary native deployment or .105 change.** [Recorded results](verified-candidate.json).

Read-only audit confirms installed MIOpen3a751adc1497 vs retained repaired native
build797335e2b713. Reused B07.11 source repair/resources, no rebuild, no source
patch reapplied. Current installed-layout RUNPATH uses $ORIGIN/rocm_sysdeps/lib;
retained build candidate still has build-tree RUNPATH. It is not ready to copy
into the active prefix without independent installed-layout packaging.

Fresh direct public solver218 calls reproduce the exact original FP16 mismatch
in BOTH daily/RCCL profiles:0.8427734375 vs FP64 reference0.8424274250864983,
error0.0003460124 exceeds unchanged0.000306262 bound. Twelve cases/profile use
six group/filter/size fixtures and FP16/FP32, actual rounded operands, independent
CPU convolution, guards and immutable inputs/weights. Old runs are intentional
negative controls, not numerical passes at the strict bound.

Isolated candidate runs the same12 direct cases/profile at the original strict
bound, all pass. Separate ordinary Torch runs12 configurations/profile: FP32,
FP16,BF16 ×groups1/3 ×dilation1/2, weighted output and ALL input/weight/bias first
gradients against CPU FP64. FP16 bounds retain B07.10 forward.002/.003,
gradients.003/.006; FP32 3e-6/5e-5, BF16 forward.003/.01, gradients.01/.02 declared
before execution. All pass. Forward repair does not claim changed backward
arithmetic, universal geometry/exceptional values, convergence or speed.

Two setup guards fail before candidate GPU numerical work: r1 ordinary launcher
correctly refuses protected LD_LIBRARY_PATH override; r2 fresh child lacks
librocm_sysdeps_zstd.so.1. r3 uses a fresh child before Torch/HIP import with an
explicit private library+sysdeps path. No global environment relaxation or
exec-replacement after HIP initialization. Both failed logs retained.

Six accepted guards (two old negative controls, two direct candidate, two Torch
candidate), two setup failures, all eight acknowledged, kernel logs unchanged,
no leftover containers. 62 local tests, Black/critical Ruff+F401/py_compile/diff
pass. Resources4CPUs/cpuset0–3/10GiB RAM/12GiB RAM+swap/512PIDs/256MiBshm,
600s recovery/540s child, PLAN-01 reservation/SH runtime lock. Native .104 prefix,
runtime descriptors and SDK/CK/ORT registrations remain unchanged. No user app
inspection, foreign stop, reboot, download or native build. Historical Python CPU
fault remains separate from fixed VM heap/D05 exit/D21 optimizer loading.

Evidence /home/nik/bc250work/evidence/plan02/{final,candidate-final,backward-final,
verified-candidate}.json and controller logs; exact phase commands/result/exit/
kernel logs under WS/logs/plan02-*, ACKs /run/plan02-* copied into archive.
Direct initial controller20838exit1, candidate21414exit1, corrected21705exit0,
backward22359exit0. Final bootafcc7be4/r2. Local kernel/artifacts/plan02/
evidence-r3.tar.gz354657B SHAfe5f48b34620cb42fb47c10deb24601686ff9b300ba4bed5b7e6c41ca2de09c2.
Originals and candidate caches kept; archive contains no model/checkpoint data.

## Exact next action

Stage a NEW installed-layout candidate, preserve existing source/build/artifact,
normalize its RUNPATH to the installed layout using the recorded packaging tool,
verify SONAME/NEEDED/public ABI and rerun these references. Audit and qualify a
coherent native cohort: installed FP32 bc250-ck pins old MIOpen bytes; SDK pins
runtime descriptor hashes; ORT registry AND immutable manifest embed the SDK
registration. Do not merely relax pins, edit one hash or replace the DSO.
Qualify retained consumers/identity refusals and reversible cohort transitions
under the documented exclusive writer lock, then ordinary .104 activation and
live rollback; deploy .105 only after matched target qualification. Keep PLAN-02
unchecked until this deployment scope passes; D26 is not the next action yet.
