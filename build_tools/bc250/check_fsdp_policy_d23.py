#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Explicit compute-only BF16 FSDP policy admission/refusal contracts."""
import torch
from torch.distributed.fsdp import MixedPrecision
from bc250_trainer_fsdp import fsdp_bf16_compute_policy, _bf16_compute_policy_matches
from check_fsdp_clipping_d22 import contracts as clipping_contracts


def contracts():
    result = clipping_contracts()
    assert _bf16_compute_policy_matches(fsdp_bf16_compute_policy())
    for p in [
        None,
        MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        ),
        MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.float32,
        ),
        MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_root_forward_inputs=True,
        ),
    ]:
        assert not _bf16_compute_policy_matches(p)
    result += [
        "explicit-FP32-parameter-reduction-buffer-BF16-compute-policy",
        "default-low-param-low-reduce-input-cast-policy-refusals",
    ]
    return result
