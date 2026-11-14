#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PLAN2-07 FP16 PEFT GPU qualification (runs on the board under the guardian).

Owned GPT2-tiny + real peft LoRA (the D28 identity), FP16 autocast working with
FP32 master/frozen/reduce + GradScaler, mirroring the frozen-helper contracts.
Parity/gradient vs an independent CPU-FP64 reference, scaler safety, root clip,
multi-update training, and bitwise save/replay. The existing frozen FSDP helpers
are untouched; no default/source/package/release change.
"""

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from peft_fp16_plan2_07 import policy  # noqa: E402

REPORT = {"checks": [], "environment": {}, "precision": dict(policy.PRECISION)}


def check(name, passed, **detail):
    REPORT["checks"].append({"name": name, "passed": bool(passed), **detail})
    if not passed:
        print(f"FP16PEFT_FAIL {name} {detail}", flush=True)
    return passed


def save(output):
    REPORT["summary"] = {
        "total": len(REPORT["checks"]),
        "passed": sum(c["passed"] for c in REPORT["checks"]),
    }
    REPORT["summary"]["all_passed"] = REPORT["summary"]["passed"] == len(
        REPORT["checks"]
    )
    output.write_text(json.dumps(REPORT, indent=2) + "\n")


def rel_l2(a, b):
    x, y = a.detach().cpu().double(), b.detach().cpu().double()
    return float((x - y).norm() / y.norm().clamp_min(1e-30))


def build_fixture(device):
    import transformers
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(policy.SEED)
    config = transformers.GPT2Config(**policy.GPT2_CONFIG)
    config._attn_implementation = "eager"
    base = transformers.GPT2LMHeadModel(config)
    lora = LoraConfig(
        r=policy.LORA_RANK,
        lora_alpha=policy.LORA_ALPHA,
        target_modules=list(policy.LORA_TARGETS),
        lora_dropout=0.0,
        bias="none",
        init_lora_weights=True,
    )
    model = get_peft_model(base, lora).to(device).train()
    model.config.use_cache = False
    ids = torch.randint(0, policy.GPT2_CONFIG["vocab_size"], (2, 16), device=device)
    return model, ids


def trainable_named(model):
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def frozen_named(model):
    return {n: p for n, p in model.named_parameters() if not p.requires_grad}


def loss_of(model, ids):
    return model(input_ids=ids, labels=ids, use_cache=False).loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    device = "cuda"
    import transformers
    import peft

    REPORT["environment"] = {
        "torch": torch.__version__,
        "hip": str(torch.version.hip),
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "gcnArchName": torch.cuda.get_device_properties(0).gcnArchName,
    }
    check(
        "gfx1013-native",
        REPORT["environment"]["gcnArchName"].split(":")[0] == "gfx1013",
    )
    save(args.output)

    model, ids = build_fixture(device)
    trainable = trainable_named(model)
    frozen = frozen_named(model)
    base_count = sum(p.numel() for p in frozen.values())
    train_count = sum(p.numel() for p in trainable.values())
    check(
        "load-param-counts",
        base_count == policy.BASE_PARAM_COUNT
        and train_count == policy.TRAINABLE_PARAM_COUNT,
        frozen=base_count,
        trainable=train_count,
    )
    check(
        "precision-contract",
        all(p.dtype == torch.float32 for p in trainable.values())
        and all(p.dtype == torch.float32 for p in frozen.values()),
        master="float32",
        working="float16-autocast",
        scaler=True,
    )
    save(args.output)

    # Independent CPU-FP64 reference (full-precision master weights).
    cpu = copy.deepcopy(model).cpu().double()
    cpu_ids = ids.cpu()
    cpu_loss = loss_of(cpu, cpu_ids)
    cpu_loss.backward()
    cpu_grads = {n: p.grad.detach().clone() for n, p in trainable_named(cpu).items()}

    # FP16 autocast forward + scaled backward, FP32 master + GradScaler.
    opt = torch.optim.AdamW(list(trainable.values()), lr=policy.LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda")
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        loss = loss_of(model, ids)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    gpu_grads = {n: p.grad.detach().clone() for n, p in trainable.items()}

    loss_rel = abs(loss.item() - cpu_loss.item()) / max(abs(cpu_loss.item()), 1e-12)
    check(
        "forward-vs-cpu-fp64",
        torch.isfinite(loss).item() and loss_rel < policy.LOSS_REL_BOUND,
        loss=loss.item(),
        reference=cpu_loss.item(),
        relative=loss_rel,
        bound=policy.LOSS_REL_BOUND,
    )

    finite = all(torch.isfinite(g).all().item() for g in gpu_grads.values())
    nonzero = any(g.abs().sum().item() > 0 for g in gpu_grads.values())
    global_rel = policy.relative_norm(
        torch.cat([gpu_grads[n].reshape(-1) for n in sorted(gpu_grads)]).tolist(),
        torch.cat([cpu_grads[n].reshape(-1) for n in sorted(cpu_grads)]).tolist(),
    )
    per_param = {n: rel_l2(gpu_grads[n], cpu_grads[n]) for n in gpu_grads}
    worst_param = max(per_param.values())
    check(
        "backward-grads-vs-cpu-fp64",
        finite
        and nonzero
        and global_rel < policy.GRAD_GLOBAL_REL_BOUND
        and worst_param < policy.GRAD_PER_PARAM_REL_BOUND,
        finite=finite,
        nonzero=nonzero,
        global_rel=global_rel,
        worst_per_param=worst_param,
        global_bound=policy.GRAD_GLOBAL_REL_BOUND,
        per_param_bound=policy.GRAD_PER_PARAM_REL_BOUND,
    )
    frozen_grads_absent = all(p.grad is None for p in frozen.values())
    check("frozen-base-no-grads", frozen_grads_absent)
    save(args.output)

    # FP16-vs-FP32 GPU relationship (FP32 is the qualified D28 path).
    fp32_model, fp32_ids = build_fixture(device)
    with torch.no_grad():
        fp32_loss = loss_of(fp32_model, fp32_ids)
    fp16_fp32_loss_rel = abs(loss.item() - fp32_loss.item()) / max(
        abs(fp32_loss.item()), 1e-12
    )
    check(
        "fp16-vs-fp32-relationship",
        fp16_fp32_loss_rel < policy.LOSS_REL_BOUND,
        loss_rel=fp16_fp32_loss_rel,
        fp16_loss=loss.item(),
        fp32_loss=fp32_loss.item(),
    )
    del fp32_model
    save(args.output)

    # Root global grad-norm clip vs CPU prediction.
    grads_list = [gpu_grads[n] for n in sorted(gpu_grads)]
    total_norm = torch.norm(torch.stack([g.norm() for g in grads_list])).item()
    clip_coef = policy.MAX_GRAD_NORM / (total_norm + 1e-6)
    expected = [g * min(1.0, clip_coef) for g in grads_list]
    torch.nn.utils.clip_grad_norm_(list(trainable.values()), policy.MAX_GRAD_NORM)
    clipped = [trainable[n].grad.detach() for n in sorted(gpu_grads)]
    clip_rel = policy.relative_norm(
        torch.cat([c.reshape(-1) for c in clipped]).tolist(),
        torch.cat([e.reshape(-1) for e in expected]).tolist(),
    )
    check(
        "root-clip-vs-cpu",
        clip_rel < policy.CLIP_NORM_REL_BOUND,
        pre_clip_norm=total_norm,
        clip_rel=clip_rel,
        bound=policy.CLIP_NORM_REL_BOUND,
    )
    save(args.output)

    # GradScaler safety: a forced inf gradient skips the step and reduces scale.
    s_model, s_ids = build_fixture(device)
    s_train = trainable_named(s_model)
    s_opt = torch.optim.AdamW(list(s_train.values()), lr=policy.LEARNING_RATE)
    s_scaler = torch.amp.GradScaler("cuda")
    s_model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        s_loss = loss_of(s_model, s_ids)
    s_scaler.scale(s_loss).backward()
    # Inject inf into the SCALED grad before unscale_: GradScaler computes
    # found_inf during unscale_(), so the skip must be triggered there.
    first = next(iter(s_train.values()))
    first.grad.fill_(float("inf"))
    s_scaler.unscale_(s_opt)
    before_w = first.detach().clone()
    scale_before = s_scaler.get_scale()
    s_scaler.step(s_opt)
    s_scaler.update()
    skipped = torch.equal(first.detach(), before_w)
    scale_reduced = s_scaler.get_scale() < scale_before
    check(
        "grad-scaler-inf-skip",
        skipped and scale_reduced,
        step_skipped=skipped,
        scale_before=scale_before,
        scale_after=s_scaler.get_scale(),
    )
    del s_model
    save(args.output)

    # Multi-update training on FP32 master adapters (fresh fixture).
    t_model, t_ids = build_fixture(device)
    t_train = trainable_named(t_model)
    t_frozen = frozen_named(t_model)
    t_opt = torch.optim.AdamW(list(t_train.values()), lr=policy.LEARNING_RATE)
    t_scaler = torch.amp.GradScaler("cuda")
    frozen_before = {n: p.detach().clone() for n, p in t_frozen.items()}
    losses = []
    for _ in range(policy.UPDATES):
        t_opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            t_loss = loss_of(t_model, t_ids)
        t_scaler.scale(t_loss).backward()
        t_scaler.unscale_(t_opt)
        torch.nn.utils.clip_grad_norm_(list(t_train.values()), policy.MAX_GRAD_NORM)
        t_scaler.step(t_opt)
        t_scaler.update()
        losses.append(t_loss.item())
    frozen_unchanged = all(
        torch.equal(t_frozen[n].detach(), frozen_before[n]) for n in t_frozen
    )
    check(
        "multi-update-training",
        all(torch.isfinite(torch.tensor(l)).item() for l in losses)
        and losses[-1] < losses[0]
        and frozen_unchanged,
        losses=losses,
        frozen_base_unchanged=frozen_unchanged,
    )
    save(args.output)

    # Bitwise save/replay: adapter + optimizer + scaler + RNG, exact continuation.
    def state_bytes(m, o):
        parts = []
        for n, p in sorted(trainable_named(m).items()):
            parts.append(
                p.detach()
                .cpu()
                .contiguous()
                .reshape(-1)
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
        for pid, st in o.state.items():
            for key in ("exp_avg", "exp_avg_sq"):
                if key in st:
                    parts.append(
                        st[key]
                        .detach()
                        .cpu()
                        .contiguous()
                        .reshape(-1)
                        .view(torch.uint8)
                        .numpy()
                        .tobytes()
                    )
        return b"".join(parts)

    t_opt.zero_grad(set_to_none=True)
    pre = state_bytes(t_model, t_opt)
    with tempfile.TemporaryDirectory() as directory:
        ckpt = Path(directory) / "peft_fp16.pt"
        torch.save(
            {
                "model": t_model.state_dict(),
                "opt": t_opt.state_dict(),
                "scaler": t_scaler.state_dict(),
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
            },
            ckpt,
        )
        ref_losses = []
        for _ in range(policy.RESUME_STEPS):
            t_opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                rl = loss_of(t_model, t_ids)
            t_scaler.scale(rl).backward()
            t_scaler.unscale_(t_opt)
            torch.nn.utils.clip_grad_norm_(list(t_train.values()), policy.MAX_GRAD_NORM)
            t_scaler.step(t_opt)
            t_scaler.update()
            ref_losses.append(rl.item())
        ref_state = state_bytes(t_model, t_opt)

        fresh, fresh_ids = build_fixture(device)
        fresh_train = trainable_named(fresh)
        fresh_opt = torch.optim.AdamW(
            list(fresh_train.values()), lr=policy.LEARNING_RATE
        )
        fresh_scaler = torch.amp.GradScaler("cuda")
        payload = torch.load(ckpt, weights_only=True, map_location="cpu")
        fresh.load_state_dict(payload["model"])
        fresh_opt.load_state_dict(payload["opt"])
        fresh_scaler.load_state_dict(payload["scaler"])
        torch.set_rng_state(payload["rng"])
        torch.cuda.set_rng_state(payload["cuda_rng"])
        restored = state_bytes(fresh, fresh_opt) == pre
        check("resume-bitwise-state", restored)
        resumed_losses = []
        for _ in range(policy.RESUME_STEPS):
            fresh_opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                sl = loss_of(fresh, fresh_ids)
            fresh_scaler.scale(sl).backward()
            fresh_scaler.unscale_(fresh_opt)
            torch.nn.utils.clip_grad_norm_(
                list(fresh_train.values()), policy.MAX_GRAD_NORM
            )
            fresh_scaler.step(fresh_opt)
            fresh_scaler.update()
            resumed_losses.append(sl.item())
        exact = (
            resumed_losses == ref_losses and state_bytes(fresh, fresh_opt) == ref_state
        )
        check(
            "resume-exact-continuation",
            exact,
            reference=ref_losses,
            resumed=resumed_losses,
        )
    save(args.output)

    REPORT["status"] = "passed" if REPORT["summary"]["all_passed"] else "failed"
    save(args.output)
    print(
        f"FP16PEFT status={REPORT['status']} "
        f"{REPORT['summary']['passed']}/{REPORT['summary']['total']}",
        flush=True,
    )
    return 0 if REPORT["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
