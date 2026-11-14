# SPDX-License-Identifier: MIT
"""Owned synthetic fixtures for PLAN2-05 (pure torch, no model packages).

These mirror the training semantics under test — causal math SDPA, frozen-base
+ trainable-LoRA adapters, embedding/LM-head/cross-entropy, and a launch-bound
LoRA residual stack with optional non-reentrant checkpointing. They are not
real models and carry no model-quality claim.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class AttentionBlock(nn.Module):
    def __init__(self, hidden, heads, kv_heads, head_dim):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.q = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o = nn.Linear(heads * head_dim, hidden, bias=False)

    def forward(self, x):
        b, s, _ = x.shape
        q = self.q(x).view(b, s, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, enable_gqa=self.heads != self.kv_heads
            )
        y = y.transpose(1, 2).reshape(b, s, self.heads * self.head_dim)
        return self.o(y)


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha=None):
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.base.weight.requires_grad_(False)
        self.base.bias.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = (alpha if alpha is not None else rank) / rank

    def forward(self, x):
        return self.base(x) + (x @ self.lora_a.t() @ self.lora_b.t()) * self.scaling


class EmbedCE(nn.Module):
    def __init__(self, vocab, hidden):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, ids, labels):
        logits = self.head(self.embed(ids))
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))


class LoRABlock(nn.Module):
    def __init__(self, width, rank):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.base = nn.Linear(width, width)
        self.base.weight.requires_grad_(False)
        self.base.bias.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.randn(rank, width) * 0.01)
        self.lora_b = nn.Parameter(torch.zeros(width, rank))
        self.scaling = 1.0

    def forward(self, x):
        y = self.norm(x)
        y = self.base(y) + (y @ self.lora_a.t() @ self.lora_b.t()) * self.scaling
        return x + F.gelu(y)


class LoRAStack(nn.Module):
    """Frozen-base + trainable-LoRA residual stack; launch-bound, owned."""

    def __init__(self, layers, width, rank, checkpointing=False):
        super().__init__()
        self.blocks = nn.ModuleList(LoRABlock(width, rank) for _ in range(layers))
        self.head = nn.Linear(width, width)
        self.head.weight.requires_grad_(False)
        self.head.bias.requires_grad_(False)
        self.checkpointing = checkpointing

    def forward(self, x):
        for block in self.blocks:
            if self.checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.head(x)

    def loss(self, x, target):
        return (self.forward(x) - target).square().mean()


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def grad_snapshot(model):
    return {
        n: p.grad.detach().clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }
