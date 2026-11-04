#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded two-host FSDP/AdamW shards and same-world checkpoint resume against CPU."""
import argparse
import copy
from datetime import timedelta
from functools import partial
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = dict(status="running", checks=[], rank=a.rank)
    try:
        torch.set_num_threads(1)
        torch.cuda.set_device(0)
        assert torch.cuda.get_device_properties(0).gcnArchName == "gfx1013:xnack-"
        dist.init_process_group("nccl", timeout=timedelta(seconds=90))
        assert dist.get_world_size() == 2 and dist.get_rank() == a.rank
        torch.manual_seed(431)
        ref = torch.nn.Sequential(
            torch.nn.Linear(16, 64), torch.nn.Tanh(), torch.nn.Linear(64, 8)
        ).double()
        total = sum(p.numel() for p in ref.parameters())

        def make():
            model = FSDP(
                copy.deepcopy(ref).float().cuda(),
                device_id=0,
                use_orig_params=True,
                auto_wrap_policy=partial(
                    size_based_auto_wrap_policy, min_num_params=32
                ),
            )
            opt = torch.optim.AdamW(
                model.parameters(), lr=0.003, weight_decay=0.01, foreach=False
            )
            return model, opt

        model, opt = make()
        refopt = torch.optim.AdamW(
            ref.parameters(), lr=0.003, weight_decay=0.01, foreach=False
        )
        local = sum(p.numel() for p in model.parameters())
        assert 0 < local < total
        count = torch.tensor(local, device="cuda", dtype=torch.int64)
        dist.all_reduce(count)
        assert int(count) == total
        r.update(full_parameter_elements=total, local_parameter_elements=local)
        r["checks"].append("actual-parameter-shards-cover-global-model")
        data = (torch.arange(32 * 16).reshape(32, 16) % 19 - 9).float() / 16
        target = (torch.arange(32 * 8).reshape(32, 8) % 13 - 6).float() / 16

        def step(index):
            opt.zero_grad()
            refopt.zero_grad()
            loss = (
                (
                    model(data[a.rank * 16 : (a.rank + 1) * 16].cuda())
                    - target[a.rank * 16 : (a.rank + 1) * 16].cuda()
                )
                .square()
                .mean()
            )
            loss.backward()
            ((ref(data.double()) - target.double()).square().mean()).backward()
            with FSDP.summon_full_params(
                model, recurse=True, writeback=False, with_grads=True
            ):
                for x, y in zip(model.parameters(), ref.parameters()):
                    torch.testing.assert_close(
                        x.grad.cpu().double(), y.grad, atol=2e-6, rtol=2e-5
                    )
            opt.step()
            refopt.step()
            with FSDP.summon_full_params(model, recurse=True, writeback=False):
                for x, y in zip(model.parameters(), ref.parameters()):
                    torch.testing.assert_close(
                        x.cpu().double(), y, atol=4e-6, rtol=4e-5
                    )
            r["checks"].append("global-reference-FSDP-gradient-AdamW-" + str(index))

        step(0)
        step(1)
        state_elements = sum(s["exp_avg"].numel() for s in opt.state.values())
        assert state_elements == local
        r["local_AdamW_exp_avg_elements"] = state_elements
        r["checks"].append("AdamW-state-sharded-with-owned-parameters")
        checkpoint = a.output.parent / "owned-fsdp-checkpoint.pt"
        torch.save(
            dict(
                rank=a.rank,
                world_size=2,
                params=[x.detach().cpu().clone() for x in model.parameters()],
                optimizer=opt.state_dict(),
                reference=ref.state_dict(),
                reference_optimizer=refopt.state_dict(),
            ),
            checkpoint,
        )
        dist.barrier()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert payload["rank"] == a.rank and payload["world_size"] == 2
        del opt, model
        model, opt = make()
        with torch.no_grad():
            for x, y in zip(model.parameters(), payload["params"]):
                assert x.shape == y.shape
                x.copy_(y.cuda())
        opt.load_state_dict(payload["optimizer"])
        ref.load_state_dict(payload["reference"])
        refopt.load_state_dict(payload["reference_optimizer"])
        r["checks"].append("same-rank-world-size-sharded-checkpoint-loaded")
        step(2)
        dist.barrier()
        dist.destroy_process_group()
        r["checks"].append("both-ranks-finish-after-resumed-update")
        r["status"] = "passed"
    except BaseException as error:
        r.update(status="failed", error=repr(error))
        raise
    finally:
        a.output.write_text(json.dumps(r, indent=2) + "\n")


if __name__ == "__main__":
    main()
