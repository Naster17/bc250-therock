#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fresh training-only D32 math with immutable external CPU qualification."""
import argparse
from datetime import timedelta
from functools import partial
import gc
import hashlib
import math
import os
from pathlib import Path
import statistics
import time
import uuid

import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
import transformers
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
import distributed_d32_capacity as qualified
from capacity_plan_d32 import CONFIG, PARAMETERS, GIB, admit
from qualification_d33 import PROOF_SHA256, MODELING_PINS, load_qualification
from distributed_d05_train import atomic_json, sha


STAT_NAMES = {
    "anon",
    "file",
    "kernel",
    "kernel_stack",
    "pagetables",
    "sock",
    "shmem",
    "slab",
    "file_mapped",
    "file_dirty",
    "file_writeback",
    "inactive_anon",
    "active_anon",
    "inactive_file",
    "active_file",
}


def sample_memory():
    result = qualified.sample_memory()
    path = Path("/sys/fs/cgroup/memory.stat")
    result["memory.stat"] = {
        s.split()[0]: int(s.split()[1])
        for s in path.read_text().splitlines()
        if s.split()[0] in STAT_NAMES
    }
    return result


class Monitor(qualified.Monitor):
    def run(self):
        while not self.stop.is_set():
            self.samples.append(
                dict(phase=self.phase, seconds=time.monotonic(), **sample_memory())
            )
            self.stop.wait(0.2)


def package_identity():
    if transformers.__version__ != "5.2.0":
        raise ValueError("Transformers version requires qualification")
    root = Path(transformers.__file__).parent
    for name, wanted in MODELING_PINS.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != wanted:
            raise ValueError("Modeling source requires qualification: " + name)


def forbidden_audit(*args, **kwargs):
    raise RuntimeError("CPU oracle/full-state audit is forbidden in the fresh worker")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["single", "ddp", "fsdp"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = dict(status="running", mode=args.mode, config=CONFIG, checks=[])
    monitor = None
    start_worker = time.perf_counter()
    try:
        physical_rank = int(os.environ["RANK"])
        proof, reference = load_qualification(args.mode, physical_rank)
        package_identity()
        if torch.__version__ != reference["torch"]:
            raise ValueError("Torch version differs from CPU qualification")
        result["qualification"] = dict(
            proof_sha256=PROOF_SHA256[physical_rank],
            archive_sha256=proof["archive_sha256"],
            source_sha256=proof["source_sha256"],
            physical_rank=physical_rank,
        )
        # These heavy audits are never needed for measurement, even accidentally.
        qualified.qualify = forbidden_audit
        FSDP.full_optim_state_dict = forbidden_audit
        FSDP.summon_full_params = forbidden_audit
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.cuda.set_device(0)
        assert torch.cuda.get_device_properties(0).gcnArchName == "gfx1013:xnack-"
        assert len(os.sched_getaffinity(0)) <= 4
        world = 1 if args.mode == "single" else 2
        if world == 2:
            dist.init_process_group("nccl", timeout=timedelta(seconds=120))
        rank = dist.get_rank() if world == 2 else 0
        plan = admit(
            args.mode, qualified.meminfo()["MemAvailable"], torch.cuda.mem_get_info()[0]
        )
        result.update(
            rank=rank,
            world=world,
            launcher_rank=physical_rank,
            torch=torch.__version__,
            plan=plan,
            source_sha256=sha(Path(__file__)),
            qualification_helper_sha256=sha(
                Path(__file__).with_name("qualification_d33.py")
            ),
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            birth=dict(
                uuid=str(uuid.uuid4()),
                pid=os.getpid(),
                start_ticks=Path("/proc/self/stat")
                .read_text()
                .split(") ", 1)[1]
                .split()[19],
            ),
            CPU_reference_executed_in_this_worker=False,
            full_state_audits_executed_in_this_worker=False,
        )
        monitor = Monitor()
        events_before = sample_memory()["events"]
        result["cgroup_events_before"] = events_before
        snapshots = []

        def snapshot(stage):
            torch.cuda.synchronize()
            snapshots.append(
                dict(
                    stage=stage,
                    allocated_bytes=torch.cuda.memory_allocated(),
                    reserved_bytes=torch.cuda.memory_reserved(),
                    device_free_bytes=torch.cuda.mem_get_info()[0],
                    **sample_memory()
                )
            )

        snapshot("before-model")
        # No CPU base duplicate is retained; the initialized model moves in place.
        model = qualified.build().cuda()
        if args.mode == "ddp":
            model = DDP(
                model,
                device_ids=[0],
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )
        elif args.mode == "fsdp":
            model = FSDP(
                model,
                device_id=0,
                use_orig_params=True,
                auto_wrap_policy=partial(
                    transformer_auto_wrap_policy, transformer_layer_cls={GPT2Block}
                ),
                limit_all_gathers=True,
            )
        opt = qualified.optimizer(model)
        local_elements = sum(p.numel() for p in model.parameters())
        assert local_elements == (
            PARAMETERS // world if args.mode == "fsdp" else PARAMETERS
        )
        snapshot("after-model-wrap")
        traffic = dict(active=False, gathers=[], reductions=[], DDP=[])
        if args.mode == "ddp":

            def hook(state, bucket):
                if state["active"]:
                    state["DDP"].append(
                        bucket.buffer().numel() * bucket.buffer().element_size()
                    )
                return allreduce_hook(None, bucket)

            model.register_comm_hook(traffic, hook)
        native_gather, native_reduce = (
            dist.all_gather_single,
            dist.reduce_scatter_single,
        )

        def gather(output, input, *args, **kwargs):
            if traffic["active"]:
                assert input.dtype == output.dtype == torch.float32
                traffic["gathers"].append(input.numel() * input.element_size())
            return native_gather(output, input, *args, **kwargs)

        def reduce(output, input, *args, **kwargs):
            if traffic["active"]:
                assert input.dtype == output.dtype == torch.float32
                traffic["reductions"].append(input.numel() * input.element_size())
            return native_reduce(output, input, *args, **kwargs)

        dist.all_gather_single, dist.reduce_scatter_single = gather, reduce

        def step(index):
            opt.zero_grad(set_to_none=True)
            value = qualified.gpu_backward(model, index, rank, world, traffic)
            (
                model.clip_grad_norm_(0.5)
                if isinstance(model, FSDP)
                else torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            )
            opt.step()
            return value

        monitor.phase = "initial-GPU-update"
        first = step(0)
        if world == 2:
            dist.all_reduce(first)
            first /= world
        assert abs(float(first) - reference["reference"]["loss"]) <= 1e-5
        result["initial_global_loss"] = float(first)
        result["checks"].extend(
            [
                "source-bound-D32-CPU-qualification",
                "fresh-worker-no-CPU-or-full-state-audits",
                "initial-GPU-loss-matches-qualified-update",
            ]
        )
        snapshot("after-initial-update")
        del first
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        snapshot("after-owned-cache-release")
        state_elements = sum(s["exp_avg"].numel() for s in opt.state.values())
        state_bytes = sum(
            v.numel() * v.element_size()
            for s in opt.state.values()
            for v in s.values()
            if isinstance(v, torch.Tensor)
        )
        assert state_elements == local_elements
        result["state"] = dict(
            full_parameter_elements=PARAMETERS,
            local_master_elements=local_elements,
            AdamW_moment_elements_each=state_elements,
            AdamW_state_bytes=state_bytes,
        )
        monitor.phase = "warmup"
        training_start = time.perf_counter()
        for index in [1, 2]:
            step(index)
        snapshot("after-warmup")
        monitor.phase = "measured-training"
        samples = []
        for offset, index in enumerate(range(3, 8)):
            if world == 2:
                dist.barrier()
            torch.cuda.synchronize()
            before = {k: len(traffic[k]) for k in ["gathers", "reductions", "DDP"]}
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            value = step(index)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            metrics = dict(
                seconds_local=seconds,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                allocated_after_bytes=torch.cuda.memory_allocated(),
                reserved_after_bytes=torch.cuda.memory_reserved(),
                device_free_bytes=torch.cuda.mem_get_info()[0],
                **sample_memory()
            )
            if world == 2:
                tensor = torch.tensor(
                    [seconds, float(value)], dtype=torch.float64, device="cuda"
                )
                dist.all_reduce(tensor[:1], op=dist.ReduceOp.MAX)
                dist.all_reduce(tensor[1:])
                metrics.update(
                    seconds_max_rank=float(tensor[0]),
                    global_loss=float(tensor[1]) / world,
                )
            else:
                metrics.update(seconds_max_rank=seconds, global_loss=float(value))
            assert (
                math.isfinite(metrics["global_loss"])
                and metrics["host_available_bytes"] > 3 * GIB
            )
            assert (
                abs(metrics["global_loss"] - reference["global_losses"][offset]) <= 1e-5
            )
            metrics["traffic_input_bytes"] = {
                k: sum(traffic[k][before[k] :]) for k in before
            }
            metrics["traffic_calls"] = {k: len(traffic[k]) - before[k] for k in before}
            samples.append(metrics)
        result["training_segment_seconds"] = time.perf_counter() - training_start
        snapshot("after-measured-training")
        result["samples"], result["snapshots"] = samples, snapshots
        result["median_update_seconds"] = statistics.median(
            s["seconds_max_rank"] for s in samples
        )
        result["checks"].extend(
            [
                "five-global-losses-match-qualified-D32-trajectory",
                "actual-master-AdamW-shard-ownership",
                "FP32-training-collective-inputs",
                "fresh-phase-allocator-cgroup-RSS-snapshots",
            ]
        )
        if world == 2:
            dist.barrier()
        result["cgroup_events_after"] = sample_memory()["events"]
        assert all(
            result["cgroup_events_after"][k] == events_before[k] == 0
            for k in ["oom", "oom_kill"]
        )
        result.update(
            status="passed", worker_seconds=time.perf_counter() - start_worker
        )
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        if monitor is not None:
            result["memory_samples"] = monitor.finish()
        if dist.is_initialized():
            dist.destroy_process_group()
        atomic_json(args.output, result)


if __name__ == "__main__":
    main()
