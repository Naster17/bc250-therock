#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CPU contract checks: label ownership, denominator, shift and analytic derivative."""
from types import SimpleNamespace

import torch
from bc250_trainer_causal import CausalLabelCollator, causal_token_loss


def contract_checks():
    checks = []
    features = [
        dict(
            input_ids=torch.tensor([1, 2, 3, 0]),
            labels=torch.tensor([1, 2, 3, 0]),
            attention_mask=torch.tensor([1, 1, 1, 0]),
        )
    ]
    original = features[0]["labels"].clone()
    batch = CausalLabelCollator()(features)
    assert torch.equal(features[0]["labels"], original)
    assert batch["labels"].tolist() == [[-100, 2, 3, -100]]
    assert (
        int(batch["labels"].ne(-100).sum())
        == int(batch["labels"][:, 1:].ne(-100).sum())
        == 2
    )
    checks.append("collator-ownership-padding-and-shift-count")
    bad = [
        dict(
            input_ids=torch.tensor([0, 1, 2]),
            labels=torch.tensor([0, 1, 2]),
            attention_mask=torch.tensor([0, 1, 1]),
        )
    ]
    try:
        CausalLabelCollator()(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("Left-padding contract accepted")
    checks.append("left-padding-refusal")
    z = torch.tensor(
        [[[0.7, -0.2, 0.4], [0.1, 0.2, -0.8], [-0.4, 0.3, 0.9], [0.2, 0.6, -0.1]]],
        requires_grad=True,
    )
    labels = torch.tensor([[-100, 2, -100, 1]])
    loss = causal_token_loss(SimpleNamespace(logits=z), labels, 7)
    source = z.detach()[0, [0, 2]]
    target = torch.tensor([2, 1])
    wanted = (
        source.logsumexp(-1) - source.gather(1, target[:, None]).squeeze(1)
    ).sum() / 7
    loss.backward()
    grad = torch.zeros_like(z)
    g = source.softmax(-1)
    g[torch.arange(2), target] -= 1
    grad[0, [0, 2]] = g / 7
    torch.testing.assert_close(loss.detach(), wanted, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(z.grad, grad, atol=1e-7, rtol=1e-6)
    checks.append("shifted-CE-loss-and-all-logit-analytic-derivatives")
    empty = z.detach().clone().requires_grad_()
    zero = causal_token_loss(
        SimpleNamespace(logits=empty), torch.full_like(labels, -100), 7
    )
    zero.backward()
    assert zero.item() == 0 and torch.equal(empty.grad, torch.zeros_like(empty))
    checks.append("all-ignored-local-positive-global-zero-loss-and-gradients")
    for count, name in [
        (None, "missing-global-count-refusal"),
        (0, "empty-global-count-refusal"),
    ]:
        try:
            causal_token_loss(SimpleNamespace(logits=z), labels, count)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid global count accepted")
        checks.append(name)
    wrong = labels.clone()
    wrong[:, 0] = 1
    try:
        causal_token_loss(SimpleNamespace(logits=z), wrong, 7)
    except ValueError:
        pass
    else:
        raise AssertionError("Unmasked first label accepted")
    checks.append("unmasked-first-label-refusal")
    return checks
