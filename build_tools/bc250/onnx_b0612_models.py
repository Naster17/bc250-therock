# SPDX-License-Identifier: MIT
"""Owned B06.1 model families reused for the new public GPU export boundary."""
import torch


class Dense(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(8, 12)
        self.last = torch.nn.Linear(12, 4)

    def forward(self, x):
        return self.last(torch.relu(self.first(x)))


class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 4, 3, padding=1)
        self.head = torch.nn.Linear(4, 3)

    def forward(self, x):
        return self.head(torch.relu(self.conv(x)).mean(dim=(-1, -2)))


class Gated(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.values = torch.nn.Linear(8, 8)
        self.gates = torch.nn.Linear(8, 8)
        self.norm = torch.nn.LayerNorm(8)
        self.head = torch.nn.Linear(8, 4)

    def forward(self, x):
        return self.head(self.norm(self.values(x) * torch.sigmoid(self.gates(x)))).mean(
            dim=1
        )


FAMILIES = (("dense", Dense, (8,)), ("cnn", CNN, (2, 8, 8)), ("gated", Gated, (5, 8)))
