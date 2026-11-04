#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Ordinary caller-context and two-host CPU exit smoke; never initialize HIP."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--literal", required=True)
a = p.parse_args()
torch.set_num_threads(1)
assert os.getuid() == 1000 and a.literal == "two words"
assert Path.cwd() == Path("/home/nik/bc250work/tools/distributed-d05")
assert not torch.cuda.is_initialized()
dist.init_process_group("gloo", init_method="env://", timeout=timedelta(seconds=45))
x = torch.tensor([dist.get_rank() + 1.0])
dist.all_reduce(x)
assert x.item() == 3.0
assert not torch.cuda.is_initialized()
dist.destroy_process_group()
a.output.write_text(
    json.dumps(
        dict(
            status="passed",
            uid=os.getuid(),
            cwd=str(Path.cwd()),
            literal=a.literal,
            HIP_initialized=False,
            CPU_sum=x.item(),
        ),
        indent=2,
    )
    + "\n"
)
