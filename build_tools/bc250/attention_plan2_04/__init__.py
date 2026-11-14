# SPDX-License-Identifier: MIT
"""PLAN2-04: performance qualification of the existing opt-in gfx1013 tiled SDPA.

No new operator and no Torch source change: the qualified object is the installed
``torch.nn.attention._gfx1013`` module. This package adds owned parity/boundary
qualification, paired math-vs-tiled long-sequence benchmarks and the orchestrator.
"""
