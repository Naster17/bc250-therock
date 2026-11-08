#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""D24 explicit mixed-precision policy contracts; D23 refusals remain intact."""
import dataclasses
import torch
from bc250_trainer_fsdp import (
    fsdp_bf16_parameter_policy,
    _bf16_parameter_policy_matches,
    fsdp_bf16_compute_policy,
)
from check_fsdp_policy_d23 import contracts as prior_contracts


def contracts():
    result = prior_contracts()
    policy = fsdp_bf16_parameter_policy()
    assert _bf16_parameter_policy_matches(policy)
    assert not _bf16_parameter_policy_matches(fsdp_bf16_compute_policy())
    for changes in [
        dict(reduce_dtype=torch.bfloat16),
        dict(buffer_dtype=torch.bfloat16),
        dict(keep_low_precision_grads=True),
        dict(cast_forward_inputs=True),
        dict(cast_root_forward_inputs=True),
        dict(param_dtype=torch.float16),
    ]:
        assert not _bf16_parameter_policy_matches(
            dataclasses.replace(policy, **changes)
        )
    return result + [
        "explicit-BF16-working-parameters-FP32-master-reduce-policy",
        "low-reduce-buffer-kept-grad-inputcast-FP16-refusals",
    ]
