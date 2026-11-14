# SPDX-License-Identifier: MIT
"""PLAN2-07: owned-fixture FP16 PEFT qualification (frozen-helper contracts).

Mirrors the D28/D30 frozen-helper precision contracts (FP32 master/frozen/reduce
+ low-precision autocast working + named LoRA + AdamW + root clip + full
save/export + bitwise replay) for the FP16 path, which the frozen FSDP helpers
do not cover (base FP16 FSDP is all-trainable; PEFT helpers are FP32/BF16). The
existing frozen helpers are untouched; no default/source/package/release change.
"""
