#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Explicit FP16 FSDP policy contracts, retaining D23/D24 policies."""
from dataclasses import replace
import torch
from bc250_trainer_fsdp import (
    fsdp_fp16_parameter_policy,
    _fp16_parameter_policy_matches,
    fsdp_bf16_parameter_policy,
)
from check_fsdp_policy_d24 import contracts as previous


def contracts():
    result = previous()
    policy = fsdp_fp16_parameter_policy()
    assert _fp16_parameter_policy_matches(policy)
    assert not _fp16_parameter_policy_matches(None)
    assert not _fp16_parameter_policy_matches(fsdp_bf16_parameter_policy())
    for changes in [
        dict(reduce_dtype=torch.float16),
        dict(buffer_dtype=torch.float16),
        dict(keep_low_precision_grads=True),
        dict(cast_forward_inputs=True),
        dict(cast_root_forward_inputs=True),
    ]:
        assert not _fp16_parameter_policy_matches(replace(policy, **changes))
    return result + [
        "explicit-FP16-working-params-FP32-master-reduce-policy",
        "default-low-reduce-buffer-kept-grad-inputcast-refusals",
    ]
