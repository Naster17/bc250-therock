"""Version-qualified native DDP checkpoint sealing and application replay checks."""

import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import socket
import uuid

import numpy as np
import torch
import torch.distributed as dist

import bc250_lora_support as base
from checkpoint_files import MANIFEST, common_inventory, file_info, validate

PINS = {
    "transformers.trainer": "33ef308cf53cd54511823f05d8888eb25d6bd567c4ef0c777b51abd65306bbef",
    "trl.trainer.sft_trainer": "ab992191b84cdc47e14c0b6baf4e6be724240da4b4d98f26341cdaba072e8e92",
    "accelerate.accelerator": "b16097a6b8fc372e580eeb35bd94f1e88c6d9dc80ab340f3d05c1daf3e64e14e",
    "bc250_lora_support": "2f24c6fedd410ee79522c771ccff66636e4dc54886db07274bd90542d8ab6824",
}
CONTEXT = {}


def digest(value):
    h = hashlib.sha256()

    def visit(x):
        if isinstance(x, torch.Tensor):
            h.update(b"tensor" + str(x.dtype).encode() + repr(tuple(x.shape)).encode())
            h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x, np.ndarray):
            h.update(b"ndarray" + str(x.dtype).encode() + repr(x.shape).encode())
            h.update(x.tobytes())
        elif isinstance(x, dict):
            h.update(b"dict{")
            for key in sorted(x, key=repr):
                visit(key)
                visit(x[key])
            h.update(b"}")
        elif isinstance(x, (list, tuple)):
            h.update(type(x).__name__.encode() + b"[")
            for item in x:
                visit(item)
            h.update(b"]")
        else:
            h.update(
                (
                    type(x).__name__
                    + ":"
                    + json.dumps(x, sort_keys=True, allow_nan=False)
                    + ";"
                ).encode()
            )

    visit(value)
    return h.hexdigest()


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def snapshot(trainer):
    params = {n: p for n, p in trainer.model.named_parameters() if p.requires_grad}
    opt = trainer.optimizer.state_dict()
    return {
        "step": trainer.state.global_step,
        "parameters": digest(params),
        "optimizer": digest(opt),
        "optimizer_state_entries": len(opt["state"]),
        "optimizer_steps": sorted(
            set(int(v["step"].item()) for v in opt["state"].values())
        ),
        "scheduler": digest(trainer.lr_scheduler.state_dict()),
        "scheduler_state": trainer.lr_scheduler.state_dict(),
        "scaler": digest(trainer.accelerator.scaler.state_dict()),
        "scaler_state": trainer.accelerator.scaler.state_dict(),
        "rng": {k: digest(v) for k, v in rng_state().items()},
        "trl_total_train_tokens": trainer._total_train_tokens,
    }


def configure(args):
    assert (
        int(os.environ.get("WORLD_SIZE", "1")) == 2
    ), "L02 requires the qualified two-rank topology"
    assert os.environ.get("BC250_LORA_BENCHMARK") != "1"
    assert (
        args.optimizer == "adamw_torch"
    ), "L02 checkpoint state is qualified for native AdamW"
    assert (
        args.num_workers == 0 and not args.packing
    ), "L02 requires workers=0 and unpacked batches"
    for name, expected in PINS.items():
        module = importlib.import_module(name)
        assert (
            hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == expected
        ), name
    action = os.environ.get("BC250_LORA_REPLAY_ACTION", "")
    assert action in ["", "seed", "resume"]
    if action:
        assert args.max_steps == 4 and args.grad_accum == 4 and args.batch_size == 2
        assert args.rank == 128 and args.dropout == 0.05 and args.precision == "fp16"
        args.validation_split = 0
        args.eval_data = None
        args.report_to = "none"
        args.save_steps = 2
        args.save_total_limit = 8
        args.epochs = 1
        assert bool(args.resume) == (action == "resume")
        assert not args.output_dir.exists(), "Use a new replay output directory"
    # Fingerprint source data/model files before GPU allocations. Paths themselves
    # may differ between boards; the file contents and training settings may not.
    model = Path(args.model).resolve()
    assert model.is_dir() and args.precision == "fp16"
    inputs = {"model": {}, "train": [], "eval": []}
    for pattern in ["*.safetensors", "*.json", "*.txt", "*.jinja"]:
        for file in sorted(model.glob(pattern)):
            inputs["model"][file.name] = file_info(file)
    for key, names in [("train", args.train_data), ("eval", args.eval_data or [])]:
        inputs[key] = [file_info(Path(name).resolve()) for name in names]
    settings = {
        k: v
        for k, v in vars(args).items()
        if k
        not in [
            "config",
            "output_dir",
            "resume",
            "allow_existing_output",
            "model",
            "train_data",
            "eval_data",
        ]
    }
    contract = {
        "world_size": 2,
        "settings": settings,
        "inputs": inputs,
        "framework_pins": PINS,
    }
    contract_hash = digest(contract)
    resume = None
    if args.resume:
        assert (
            args.resume != "latest"
        ), "Use an explicit complete sealed checkpoint on each host"
        resume = validate(Path(args.resume), int(os.environ["RANK"]))
        assert (
            resume["contract_sha256"] == contract_hash
        ), "Training settings, data or model changed"
    CONTEXT.update(
        action=action,
        contract=contract,
        contract_sha256=contract_hash,
        resume=resume,
        output=args.output_dir,
    )
    return bool(action)


def configure_trainer(trainer):
    base.configure_trainer(trainer)
    if CONTEXT["action"] == "seed":
        s = trainer.accelerator.scaler.state_dict()
        s["growth_interval"] = 1  # exercise saved/restored non-default scaler states
        trainer.accelerator.scaler.load_state_dict(s)
    if CONTEXT["resume"]:
        rank = os.environ["RANK"]
        trainer._total_train_tokens = CONTEXT["resume"]["ranks"][rank]["state"][
            "trl_total_train_tokens"
        ]


class Audit(base.Audit):
    def on_train_begin(self, args, state, control, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        self.proof = {
            "status": "running",
            "action": CONTEXT["action"],
            "rank": self.rank,
            "pid": os.getpid(),
            "start_ticks": Path("/proc/self/stat").read_text().split()[21],
            "container_hostname": socket.gethostname(),
            "instance_uuid": str(uuid.uuid4()),
            "initial": snapshot(self.trainer),
            "steps": [],
            "checkpoints": [],
        }
        if CONTEXT["resume"]:
            saved = CONTEXT["resume"]["ranks"][str(self.rank)]["state"]
            current = self.proof["initial"]
            for key in [
                "step",
                "parameters",
                "optimizer",
                "optimizer_state_entries",
                "optimizer_steps",
                "scheduler",
                "scaler",
                "trl_total_train_tokens",
            ]:
                assert current[key] == saved[key], ("Loaded state differs", key)
            if CONTEXT["action"]:
                # Native Trainer restores all rank RNGs after iterator skipping,
                # later than on_train_begin. Poison now to test that restoration.
                random.seed(1773)
                np.random.seed(1773)
                torch.manual_seed(1773)
                torch.cuda.manual_seed_all(1773)
        self.run_id = [str(uuid.uuid4()) if self.rank == 0 else None]
        dist.broadcast_object_list(self.run_id, src=0)
        self.persist()

    def persist(self):
        (self.output / f"replay-rank{self.rank}.json").write_text(
            json.dumps(self.proof, indent=2)
        )

    def on_step_begin(self, args, state, control, **kwargs):
        super().on_step_begin(args, state, control, **kwargs)
        if CONTEXT["action"]:
            self.before = snapshot(self.trainer)
            if CONTEXT["resume"] and not self.proof["steps"]:
                expected = CONTEXT["resume"]["ranks"][str(self.rank)]["state"]
                assert (
                    self.before["rng"] == expected["rng"]
                ), "Native rank RNG restore failed"

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        super().on_pre_optimizer_step(args, state, control, **kwargs)
        if CONTEXT["action"]:
            self.gradients = digest({n: p.grad for n, p in self.params})

    def on_step_end(self, args, state, control, **kwargs):
        super().on_step_end(args, state, control, **kwargs)
        if CONTEXT["action"]:
            skipped = self.rows[-1]["optimizer_skipped"]
            assert skipped == (not self.rows[-1]["finite_gradients"])
            after = snapshot(self.trainer)
            if skipped:
                for key in ["parameters", "optimizer", "scheduler"]:
                    assert after[key] == self.before[key], (
                        "Overflow updated state",
                        key,
                    )
                previous = self.before["scaler_state"]
                assert (
                    after["scaler_state"]["scale"]
                    == previous["scale"] * previous["backoff_factor"]
                )
            self.proof["steps"].append(
                {
                    "step": state.global_step,
                    "optimizer_skipped": skipped,
                    "finite_gradients": self.rows[-1]["finite_gradients"],
                    "before": self.before,
                    "gradients": self.gradients,
                    "after": after,
                    "data": {
                        k: self.rows[-1][k]
                        for k in [
                            "examples",
                            "targets",
                            "input_tokens",
                            "padded_tokens",
                            "example_sha256",
                        ]
                    },
                }
            )
            control.should_save = CONTEXT["action"] == "seed" and state.global_step == 2
            self.persist()

    def on_epoch_end(self, args, state, control, **kwargs):
        super().on_epoch_end(args, state, control, **kwargs)
        if CONTEXT["action"]:
            control.should_save = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        super().on_log(args, state, control, logs=logs, **kwargs)
        if CONTEXT["action"] and self.proof["steps"] and logs and "loss" in logs:
            self.proof["steps"][-1]["metrics"] = {
                k: (
                    str(logs[k])
                    if isinstance(logs[k], float) and not math.isfinite(logs[k])
                    else logs[k]
                )
                for k in [
                    "loss",
                    "grad_norm",
                    "learning_rate",
                    "entropy",
                    "mean_token_accuracy",
                    "num_tokens",
                ]
            }
            self.persist()

    def on_save(self, args, state, control, **kwargs):
        checkpoint = self.output / f"checkpoint-{state.global_step}"
        own = {
            "rank": self.rank,
            "state": snapshot(self.trainer),
            "rng_file": file_info(checkpoint / f"rng_state_{self.rank}.pth"),
        }
        own["payload"] = common_inventory(checkpoint) if self.rank == 0 else None
        ranks = [None, None]
        dist.all_gather_object(ranks, own)
        for item in ranks:
            for key in ["parameters", "optimizer", "scheduler", "scaler"]:
                assert item["state"][key] == ranks[0]["state"][key], key
        manifest = {
            "format": "bc250-lora-ddp-v1",
            "world_size": 2,
            "step": state.global_step,
            "checkpoint_id": self.run_id[0] + f"-{state.global_step}",
            "contract_sha256": CONTEXT["contract_sha256"],
            "contract": CONTEXT["contract"],
            "payload": ranks[0]["payload"],
            "ranks": {
                str(r["rank"]): {k: v for k, v in r.items() if k != "payload"}
                for r in ranks
            },
        }
        with (checkpoint / MANIFEST).open("x") as stream:
            json.dump(manifest, stream, indent=2)
        self.proof["checkpoints"].append(
            {"step": state.global_step, "id": manifest["checkpoint_id"]}
        )
        self.persist()

    def on_train_end(self, args, state, control, **kwargs):
        super().on_train_end(args, state, control, **kwargs)
        if CONTEXT["action"] == "resume":
            reference = json.loads(
                (
                    Path(os.environ["BC250_LORA_REPLAY_REFERENCE"])
                    / f"replay-rank{self.rank}.json"
                ).read_text()
            )
            assert (
                reference["status"] == "passed"
                and reference["instance_uuid"] != self.proof["instance_uuid"]
            )
            # Host-network containers retain the board hostname; namespace PIDs
            # may be reused. Kernel start times identify fresh workers.
            assert reference["start_ticks"] != self.proof["start_ticks"]
            expected = [
                x for x in reference["steps"] if x["step"] > CONTEXT["resume"]["step"]
            ]
            assert (
                self.proof["steps"] == expected
            ), "Fresh-process update transcript differs"
        self.proof["status"] = "passed"
        self.persist()
