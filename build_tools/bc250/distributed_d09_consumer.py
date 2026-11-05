#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Owned offline Transformers DDP reference and ordinary caller CPU context fixture."""
import argparse
from contextlib import nullcontext
import copy
from datetime import timedelta
import errno
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--rw", type=Path, required=True)
    p.add_argument("--expected-cwd", type=Path, required=True)
    p.add_argument("--argument", required=True)
    p.add_argument("--cpu-only", action="store_true")
    a = p.parse_args()
    result = dict(status="running", checks=[])
    try:
        assert (
            os.getuid() == 1000
            and Path.cwd() == a.expected_cwd
            and a.argument == "two words"
        )
        assert os.environ["BC250_D09_ORDER"] == "command override"
        assert os.environ["BC250_D09_CALLER"] == "caller with spaces"
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        data = json.loads(a.data.read_text())
        assert data["fixture"] == "owned-distributed-d09"
        try:
            (a.data.parent / "must-not-write").write_text("blocked")
        except OSError as error:
            assert error.errno == errno.EROFS
        else:
            raise AssertionError("Read-only bind became writable")
        result["checks"].append("uid-cwd-spaced-argv-env-precedence-RO-mount")
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        backend = "gloo" if a.cpu_only else "nccl"
        dist.init_process_group(
            backend, init_method="env://", timeout=timedelta(seconds=90)
        )
        rank, world = dist.get_rank(), dist.get_world_size()
        assert world == 2
        marker = a.rw / f"rank-{rank}.txt"
        marker.write_text("owned writable mount\n")
        assert marker.read_text() == "owned writable mount\n"
        result.update(
            rank=rank,
            world=world,
            torch=torch.__version__,
            torch_path=torch.__file__,
            cwd=str(Path.cwd()),
            uid=os.getuid(),
        )
        result["checks"].append("RW-bind-owned-write-and-two-host-process-group")
        if a.cpu_only:
            import bc250_distributed_identity as identity

            identity.check_runtime()
            with patch.object(
                torch, "__file__", "/tmp/unqualified-consumer/torch/__init__.py"
            ):
                try:
                    identity.check_runtime()
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("Torch shadow package admitted")
            pin = json.loads(
                (Path(identity.__file__).parent / "torch-identity.json").read_text()
            )
            for field in ["version", "native-hash"]:
                bad = copy.deepcopy(pin)
                if field == "version":
                    bad["version"] = "unqualified"
                else:
                    bad["sha256"]["torch/lib/libtorch_hip.so"] = "0" * 64
                with patch.object(identity.json, "loads", return_value=bad):
                    try:
                        identity.check_runtime()
                    except RuntimeError:
                        pass
                    else:
                        raise AssertionError("Wrong native identity admitted")
            assert not torch.cuda.is_initialized()
            total = torch.tensor(rank + 1)
            dist.all_reduce(total)
            assert total.item() == 3
            result["torch_cuda_initialized"] = torch.cuda.is_initialized()
            result["checks"].append(
                "CPU-Gloo-sum-three-and-three-runtime-identity-refusals"
            )
        else:
            from transformers import BertConfig, BertForMaskedLM
            import transformers, peft, accelerate

            torch.cuda.set_device(0)
            assert torch.cuda.get_device_properties(0).gcnArchName == "gfx1013:xnack-"
            config = BertConfig(
                vocab_size=97,
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=64,
                max_position_embeddings=32,
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
                pad_token_id=0,
                attn_implementation="eager",
            )
            torch.manual_seed(773)
            reference = BertForMaskedLM(config).float().double()
            model = DDP(
                copy.deepcopy(reference).float().cuda(),
                device_ids=[0],
                forward_sync_buffers=False,
            )
            opt = torch.optim.AdamW(
                model.parameters(), lr=0.001, eps=1e-5, weight_decay=0.01, foreach=False
            )
            refopt = torch.optim.AdamW(
                reference.parameters(),
                lr=0.001,
                eps=1e-5,
                weight_decay=0.01,
                foreach=False,
            )
            ids = torch.tensor(data["input_ids"], dtype=torch.long)
            labels = (ids * 3 + 7) % 97
            assert ids.shape == (8, 12)
            result.update(
                transformers=transformers.__version__,
                transformers_path=transformers.__file__,
                peft=peft.__version__,
                accelerate=accelerate.__version__,
                parameter_elements=sum(v.numel() for v in model.parameters()),
                config=config.to_dict(),
            )
            for step in range(2):
                opt.zero_grad(set_to_none=True)
                refopt.zero_grad(set_to_none=True)
                local_loss = torch.zeros((), device="cuda")
                for micro in range(2):
                    start = rank * 4 + micro * 2
                    with model.no_sync() if micro == 0 else nullcontext():
                        loss = (
                            model(
                                input_ids=ids[start : start + 2].cuda(),
                                labels=((labels + step) % 97)[start : start + 2].cuda(),
                            ).loss
                            / 2
                        )
                        loss.backward()
                    local_loss += loss.detach()
                expected = reference(input_ids=ids, labels=(labels + step) % 97).loss
                expected.backward()
                dist.all_reduce(local_loss)
                local_loss /= world
                torch.testing.assert_close(
                    local_loss.cpu().double(), expected, atol=3e-6, rtol=5e-5
                )
                for actual, wanted in zip(model.parameters(), reference.parameters()):
                    assert actual.grad is not None and wanted.grad is not None
                    torch.testing.assert_close(
                        actual.grad.cpu().double(), wanted.grad, atol=3e-6, rtol=5e-4
                    )
                opt.step()
                refopt.step()
                for actual, wanted in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(
                        actual.detach().cpu().double(), wanted, atol=4e-6, rtol=5e-5
                    )
                result["checks"].append(
                    f"offline-Bert-global-loss-all-gradients-AdamW-{step}"
                )
            result["checks"].append(
                "frozen-Transformers-peft-accelerate-imports-native-Torch-preserved"
            )
        dist.barrier()
        dist.destroy_process_group()
        result["status"] = "passed"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        a.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
