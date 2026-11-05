#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CPU precision-scope ownership, restoration, and refusal contracts."""
import torch
from bc250_eval_precision import fp32_eval_head


def contracts():
    torch.manual_seed(2020)
    head = torch.nn.Sequential(
        torch.nn.Linear(8, 8), torch.nn.GELU(), torch.nn.Linear(8, 5)
    ).eval()
    x = torch.randn(2, 8)
    keys = list(head.state_dict())
    ids = [id(p) for p in head.parameters()]
    before = {k: v.clone() for k, v in head.state_dict().items()}
    original = head.forward
    with torch.no_grad():
        ref = head(x)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        assert head(x).dtype == torch.bfloat16
        with fp32_eval_head(head):
            assert head(x).dtype == torch.float32 and torch.equal(head(x), ref)
            try:
                with fp32_eval_head(head):
                    pass
            except ValueError:
                pass
            else:
                raise AssertionError("Nested scope accepted")
        assert head(x).dtype == torch.bfloat16
    assert "forward" not in head.__dict__ and head.forward == original
    assert keys == list(head.state_dict()) and ids == [id(p) for p in head.parameters()]
    assert all(torch.equal(v, before[k]) for k, v in head.state_dict().items())
    try:
        with fp32_eval_head(head):
            raise RuntimeError("owned-probe")
    except RuntimeError:
        pass
    assert "forward" not in head.__dict__ and not hasattr(
        head, "_bc250_fp32_eval_scope"
    )
    for kind in ["training", "bf16-weights", "grad", "bad-input"]:
        candidate = torch.nn.Linear(8, 5).eval()
        if kind == "training":
            candidate.train()
        if kind == "bf16-weights":
            candidate = candidate.bfloat16()
        try:
            with fp32_eval_head(candidate):
                if kind == "grad":
                    candidate(x)
                elif kind == "bad-input":
                    with torch.no_grad():
                        candidate(x, x)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid head/input accepted")
    # Preserve a preexisting user instance forward exactly, without class patching.
    custom = torch.nn.Linear(8, 5).eval()
    saved = custom.forward
    custom.forward = lambda value: saved(value)
    owned = custom.__dict__["forward"]
    with torch.no_grad(), fp32_eval_head(custom):
        assert custom(x).dtype == torch.float32
    assert custom.__dict__["forward"] is owned
    return [
        "FP32-scope-and-native-AMP-restoration",
        "parameter-values-identities-and-state-keys-unchanged",
        "exception-and-custom-forward-restoration",
        "nested-training-weights-grad-input-refusals",
    ]
