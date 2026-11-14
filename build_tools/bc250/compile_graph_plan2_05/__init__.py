# SPDX-License-Identifier: MIT
"""PLAN2-05: torch.compile expansion + graph-capture integration qualification.

Qualifies owned fixtures only; no app entry-point, Torch source, default,
package or release change. Part A expands compiled-training coverage to
attention/LoRA/embedding+CE/dynamic shapes; Part B qualifies HIP graph capture
of a LoRA-pattern training step (fwd+bwd+optimizer, checkpointing, fresh-process
recapture) and ships a reusable opt-in capture helper.
"""
