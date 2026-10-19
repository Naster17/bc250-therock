import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

from .formatting import FORMAT_VERSION, IGNORE_INDEX, encode_record


def training_order(rows, longest_first=False):
    order = list(range(len(rows)))
    random.shuffle(order)
    if longest_first:
        order.sort(key=lambda index: len(rows[index]["input_ids"]), reverse=True)
    return order


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Optional local-only Base-model LoRA training, batch size 1. No chat template or full tuning.",
        epilog="Install requirements-train.txt in your training environment. --preflight loads only local "
               "config/tokenizer, not weights. Overlength records are rejected, never silently truncated. "
               "Qwen3.5 needs a compatible Transformers text-only loader; hardware/kernel support is not guaranteed.")
    parser.add_argument("--model", required=True, type=Path, help="Downloaded local model directory, not a Hub ID")
    parser.add_argument("--tokenizer", type=Path, help="Optional local fast tokenizer directory")
    parser.add_argument("--dataset", required=True, type=Path, help="Dataset directory with train.jsonl and val.jsonl")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", type=Path, help="Trusted checkpoint directory; contains pickle training state")
    parser.add_argument("--preflight", action="store_true", help="Validate and tokenize data without loading weights or training")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, help="Cap optimizer steps within the epoch budget")
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--targets", default="auto-linear", help="auto-linear: all actual Linear modules in language decoder "
                        "layers, excluding vision and lm_head; or comma-separated leaf names such as q_proj,v_proj "
                        "(incomplete coverage of hybrid Qwen3.5 linear-attention layers)")
    parser.add_argument("--save-every", type=int, default=100, help="Optimizer-step checkpoint interval; 0 disables periodic saves")
    parser.add_argument("--eval-every", type=int, default=100, help="Optimizer-step evaluation interval; 0 disables periodic evaluation")
    parser.add_argument("--eval-subset", type=int, default=0, help="Fixed seeded validation subset; 0 evaluates all records")
    parser.add_argument("--schedule", choices=("linear", "cosine"), default="linear")
    parser.add_argument("--longest-first", action="store_true", help="Test longest sequences first, intended for bounded smoke runs")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"), help="ROCm also uses cuda")
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp32"))
    parser.add_argument("--sdpa", default="math", choices=("math", "default"))
    parser.add_argument("--attention-backend", default="math", choices=("math", "tiled"), help="Explicit BC-250 tiled attention; math keeps existing resume fingerprints")
    args = parser.parse_args(argv)
    for key in ("epochs", "accumulation_steps", "rank", "alpha"):
        if getattr(args, key) < 1:
            parser.error(key.replace("_", "-") + " must be positive")
    if args.max_seq_length < 2 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("max-seq-length must be at least 2 and max-steps must be positive")
    if min(args.save_every, args.eval_every, args.warmup_steps, args.eval_subset) < 0:
        parser.error("Intervals and warmup-steps must be nonnegative")
    for key in ("learning_rate", "weight_decay", "max_grad_norm"):
        value = getattr(args, key)
        if not math.isfinite(value) or value < 0 or (key != "weight_decay" and value == 0):
            parser.error(key.replace("_", "-") + " must be finite and positive (weight decay may be zero)")
    if not args.model.is_dir() or (args.tokenizer is not None and not args.tokenizer.is_dir()):
        parser.error("model and tokenizer must be existing local directories")
    if not args.targets.strip():
        parser.error("targets must not be empty")
    try:
        import transformers
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as error:
        parser.error(f"Optional training dependencies unavailable: {error}. Install requirements-train.txt")

    config = AutoConfig.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model, use_fast=True,
                                            local_files_only=True, trust_remote_code=False)
    text_config = getattr(config, "text_config", config)
    position_limit = getattr(text_config, "max_position_embeddings", None)
    if position_limit is not None and args.max_seq_length > position_limit:
        parser.error(f"max-seq-length exceeds model position limit {position_limit}")
    tokenizer_sha256 = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
    manifest_path = args.dataset / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("quality", {}).get("errors"):
            parser.error("Dataset manifest reports quality errors; rebuild before training")
        tokenization = manifest.get("tokenization")
        if tokenization is not None:
            if not isinstance(tokenization, dict):
                parser.error("Dataset manifest tokenization must be an object")
            for key, expected in (("tokenizer_sha256", tokenizer_sha256), ("format_version", FORMAT_VERSION)):
                if key in tokenization and (type(tokenization[key]) is not type(expected) or tokenization[key] != expected):
                    parser.error(f"Dataset manifest tokenization {key} mismatch")
            if "max_seq_length" in tokenization:
                build_limit = tokenization["max_seq_length"]
                if type(build_limit) is not int or build_limit < 2:
                    parser.error("Dataset manifest tokenization max_seq_length must be an integer of at least 2")
                if build_limit > args.max_seq_length:
                    parser.error("Dataset manifest tokenization max_seq_length exceeds runtime max-seq-length")
    data = {}
    data_hash = hashlib.sha256()
    for split in ("train", "val"):
        path = args.dataset / f"{split}.jsonl"
        data_hash.update(split.encode())
        rows = []
        split_hash = hashlib.sha256()
        with path.open("rb") as stream:
            for line_number, line in enumerate(stream, 1):
                data_hash.update(line)
                split_hash.update(line)
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    encoded = encode_record(record, tokenizer, args.max_seq_length)
                    if record.get("token_count", len(encoded["input_ids"])) != len(encoded["input_ids"]):
                        raise ValueError("Stored token_count differs from actual tokenizer output")
                    supervised = sum(value != IGNORE_INDEX for value in encoded["labels"][1:])
                    if record.get("supervised_tokens", supervised) != supervised:
                        raise ValueError("Stored supervised_tokens differs from actual label mask")
                    rows.append(encoded)
                except (ValueError, TypeError, KeyError) as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
        data[split] = rows
        expected = manifest.get("files_sha256", {}).get(split+".jsonl")
        if expected is not None and expected != split_hash.hexdigest():
            parser.error(f"Dataset checksum mismatch for {split}.jsonl; transfer corruption or untracked edits")
    if not data["train"]:
        parser.error("Training split has no records")
    total_steps = args.epochs * math.ceil(len(data["train"]) / args.accumulation_steps)
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    model_files = [(str(path.relative_to(args.model)), path.stat().st_size, path.stat().st_mtime_ns)
                   for path in sorted(args.model.rglob("*")) if path.is_file()]
    run_config = {key: str(value.resolve()) if isinstance(value, Path) else value
                  for key, value in vars(args).items() if key not in ("output", "resume", "preflight", "dataset")}
    if args.attention_backend == "math":
        run_config.pop("attention_backend")
    run_config.update(format_version=FORMAT_VERSION, data_sha256=data_hash.hexdigest(),
                      model_config=config.to_dict(), model_files=model_files, total_steps=total_steps,
                      tokenizer_sha256=tokenizer_sha256,
                       transformers_version=transformers.__version__)
    run_config["runtime_packages"] = {}
    for package in ("torch", "peft", "accelerate", "safetensors", "tokenizers", "huggingface-hub"):
        try:
            run_config["runtime_packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            run_config["runtime_packages"][package] = None
    run_config = json.loads(json.dumps(run_config, sort_keys=True))
    fingerprint = hashlib.sha256(json.dumps(run_config, sort_keys=True).encode()).hexdigest()
    print(json.dumps({"architecture": getattr(config, "architectures", None), "model_type": config.model_type,
                      "text_model_type": text_config.model_type, "fingerprint": fingerprint,
                      "splits": {split: {"records": len(rows),
                                          "target_tokens": sum(sum(x != IGNORE_INDEX for x in row["labels"][1:]) for row in rows),
                                          "longest": max((len(row["input_ids"]) for row in rows), default=0)}
                                 for split, rows in data.items()}, "optimizer_steps": total_steps}, indent=2))
    from . import bc250_checkpoint_provenance as provenance
    saved_provenance = None
    saved = None
    if args.resume is not None:
        saved = json.loads((args.resume / "run.json").read_text())
        if (saved.get("fingerprint") != fingerprint or saved.get("config") != run_config
                or set(saved) - {"fingerprint", "config", provenance.MARKER}):
            parser.error("Resume config/data/model/tokenizer fingerprint mismatch; use the original run settings")
    if args.resume is not None:
        try:
            saved_provenance = provenance.read_checkpoint(args.resume, saved, fingerprint)
        except ValueError as error:
            parser.error(str(error))
    if args.preflight:
        print("Preflight complete. No weights loaded, output written, or training started. Targets verified only when weights load.")
        return

    try:
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
    except ImportError as error:
        parser.error(f"Optional trainer unavailable: {error}. Install requirements-train.txt")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA/ROCm device unavailable; run on the training machine or use --preflight")
    if args.device == "cuda" and args.dtype == "bf16" and not torch.cuda.is_bf16_supported():
        parser.error("Device does not report bf16 support; choose --dtype fp32 if appropriate")
    if args.output.exists() and any(args.output.iterdir()) and args.resume is None:
        parser.error("Output is not empty; use a new directory or --resume")
    if args.device == "cuda":
        properties = torch.cuda.get_device_properties(0)
        print(json.dumps({"torch": torch.__version__, "device": properties.name,
                          "architecture": getattr(properties, "gcnArchName", None),
                          "device_memory_bytes": properties.total_memory,
                          "runtime_packages": run_config["runtime_packages"]}), flush=True)
    try:
        runtime_provenance = provenance.capture(torch, __file__, fingerprint, args.device)
        if args.resume is not None:
            provenance.check_resume(saved_provenance, runtime_provenance)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    loader = AutoModelForCausalLM
    load_options = {}
    if text_config.model_type == "qwen3_5_text":
        loader = getattr(transformers, "Qwen3_5ForCausalLM", None)
        if loader is None:
            parser.error("This Transformers build has no Qwen3_5ForCausalLM text-only loader; upgrade on the training machine")
        if text_config is not config:
            load_options["key_mapping"] = {r"^model.language_model\.": "model."}
    elif text_config is not config:
        parser.error("Unsupported composite architecture; a verified text-only loader is required")
    attention_implementation = "sdpa"
    if args.attention_backend == "tiled":
        if args.device != "cuda":
            parser.error("Tiled attention requires the native gfx1013 GPU")
        from torch.nn.attention._gfx1013 import register_transformers_attention
        register_transformers_attention()
        attention_implementation = "bc250_tiled"
    print("Loading local text-only base weights; no vision encoder will be trained", flush=True)
    model, loading = loader.from_pretrained(args.model, config=text_config, dtype=dtype,
                                           attn_implementation=attention_implementation, local_files_only=True,
                                           trust_remote_code=False, output_loading_info=True, **load_options)
    if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise RuntimeError(f"Text-only weights did not load completely: {loading}")
    if any(part in {"visual", "vision_model", "vision_tower"} for name, _ in model.named_modules() for part in name.split(".")):
        raise ValueError("Loader retained vision components; refusing unverified composite training")
    requested = None if args.targets == "auto-linear" else set(args.targets.split(","))
    targets = []
    for name, module in model.named_modules():
        parts = name.split(".")
        if "layers" not in parts or any("vision" in part or "visual" in part for part in parts):
            continue
        layer_index = parts.index("layers") + 1
        if layer_index >= len(parts) or not parts[layer_index].isdigit():
            continue
        if isinstance(module, torch.nn.Linear) and (requested is None or parts[-1] in requested):
            targets.append(name)
    if not targets or (requested is not None and requested - {name.split(".")[-1] for name in targets}):
        parser.error("Requested LoRA targets were not found in actual language decoder Linear modules")
    if requested is not None and any("linear_attn" in name for name, _ in model.named_modules()):
        print("Explicit target list: verify hybrid linear-attention coverage; q_proj,v_proj alone omits gated-delta layers.")
    print(json.dumps({"lora_targets": targets}, indent=2))
    model.config.use_cache = False
    if args.resume is None:
        model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0,
                                               target_modules=targets, bias="none", task_type="CAUSAL_LM"))
    else:
        adapter_config = LoraConfig.from_pretrained(args.resume, local_files_only=True)
        from peft.tuners.tuners_utils import check_target_module_exists
        resolved_targets = {name for name, _ in model.named_modules()
                            if check_target_module_exists(adapter_config, name)}
        if resolved_targets != set(targets) or adapter_config.r != args.rank or adapter_config.lora_alpha != args.alpha:
            parser.error("Checkpoint adapter target/rank/alpha mismatch")
        model = PeftModel.from_pretrained(model, args.resume, is_trainable=True, local_files_only=True)
    model.to(args.device)
    if text_config.model_type == "qwen3_5_text" and args.device == "cuda":
        enabled = os.environ.get("BC250_QWEN35_TRAINING_OPS", "1")
        if enabled not in {"0", "1"}:
            raise ValueError("BC250_QWEN35_TRAINING_OPS must be 0 or 1")
        if enabled == "1":
            from .bc250_ops import install
            install(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    schedule = get_cosine_schedule_with_warmup if args.schedule == "cosine" else get_linear_schedule_with_warmup
    scheduler = schedule(optimizer, args.warmup_steps, total_steps)
    epoch, position, step = 0, 0, 0
    order = training_order(data["train"], args.longest_first)
    if args.resume is not None:
        state = torch.load(args.resume / "training.pt", map_location="cpu", weights_only=False)
        if state.get(provenance.MARKER) != (saved or {}).get(provenance.MARKER):
            parser.error("Training state/runtime provenance binding mismatch")
        if state["fingerprint"] != fingerprint:
            parser.error("Training state fingerprint mismatch")
        epoch, position, step, order = (state[key] for key in ("epoch", "position", "step", "order"))
        if sorted(order) != list(range(len(data["train"]))) or not 0 <= position <= len(order) or not 0 <= epoch <= args.epochs or not 0 <= step <= total_steps:
            parser.error("Invalid checkpoint data progress")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None:
            if not torch.cuda.is_available() or len(state["cuda_rng"]) != torch.cuda.device_count():
                parser.error("Resume CUDA/ROCm RNG device count mismatch")
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    args.output.mkdir(parents=True, exist_ok=True)
    eval_rows = data["val"]
    if args.eval_subset and len(eval_rows) > args.eval_subset:
        eval_rows = (sorted(eval_rows, key=lambda row: len(row["input_ids"]), reverse=True)[:args.eval_subset]
                     if args.longest_first else random.Random(args.seed).sample(eval_rows, args.eval_subset))

    def batch(row):
        return {key: torch.tensor([value], dtype=torch.long, device=args.device) for key, value in row.items()}

    def evaluate():
        if not data["val"]:
            print("Evaluation skipped: validation split is empty")
            return
        model.eval()
        weighted_loss, tokens = 0.0, 0
        with torch.no_grad():
            for row in eval_rows:
                count = sum(label != IGNORE_INDEX for label in row["labels"][1:])
                with sdpa_kernel(SDPBackend.MATH) if args.sdpa == "math" else nullcontext():
                    with torch.autocast(args.device, dtype=dtype, enabled=args.dtype == "bf16"):
                        loss = model(**batch(row)).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite evaluation loss")
                weighted_loss += loss.item() * count
                tokens += count
        eval_loss = weighted_loss / tokens
        if os.environ.get("REPLICA_LOG_FORMAT", "pretty").lower() != "json":
            print(f"[eval {step:03d}/{total_steps}] eval_loss: {eval_loss:.4f} | tokens: {tokens} | records: {len(eval_rows)}/{len(data['val'])}", flush=True)
        else:
            print(json.dumps({"step": step, "eval_loss": eval_loss, "eval_target_tokens": tokens,
                              "eval_records": len(eval_rows), "eval_total_records": len(data["val"])}), flush=True)
        model.train()

    def save():
        destination = args.output / f"checkpoint-{step:08d}"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite checkpoint: {destination}")
        staging = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=args.output))
        model.save_pretrained(staging, safe_serialization=True)
        tokenizer.save_pretrained(staging)
        provenance_sha256 = provenance.write_checkpoint(staging, runtime_provenance)
        (staging / "run.json").write_text(json.dumps({"fingerprint": fingerprint, "config": run_config, provenance.MARKER: provenance_sha256}, indent=2) + "\n")
        torch.save({"fingerprint": fingerprint, provenance.MARKER: provenance_sha256, "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "python_rng": random.getstate(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "epoch": epoch, "position": position, "step": step, "order": order}, staging / "training.pt")
        os.rename(staging, destination)
        if os.environ.get("REPLICA_LOG_FORMAT", "pretty").lower() != "json":
            print(f"[checkpoint] Saved {destination.name}", flush=True)
        else:
            print(f"Saved {destination}")

    last_saved = step if args.resume is not None else None
    last_evaluated = None
    model.train()
    optimizer.zero_grad(set_to_none=True)
    session_start = time.perf_counter()
    session_tokens = 0
    session_target_tokens = 0
    session_steps = 0
    use_pretty_log = os.environ.get("REPLICA_LOG_FORMAT", "pretty").lower() != "json"

    while epoch < args.epochs and step < total_steps:
        if position == len(order):
            epoch += 1
            if epoch == args.epochs:
                break
            order = training_order(data["train"], args.longest_first)
            position = 0
        indices = order[position:position + args.accumulation_steps]
        started = time.perf_counter()
        token_counts = [sum(label != IGNORE_INDEX for label in data["train"][index]["labels"][1:]) for index in indices]
        total_tokens = sum(token_counts)
        training_loss = 0.0
        for index, count in zip(indices, token_counts):
            with sdpa_kernel(SDPBackend.MATH) if args.sdpa == "math" else nullcontext():
                with torch.autocast(args.device, dtype=dtype, enabled=args.dtype == "bf16"):
                    loss = model(**batch(data["train"][index])).loss
                    scaled_loss = loss * (count / total_tokens)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss; no checkpoint written for this partial step")
                scaled_loss.backward()
            training_loss += loss.detach().item() * count / total_tokens
        torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        position += len(indices)
        step += 1
        step_seconds = time.perf_counter() - started
        input_tokens = sum(len(data["train"][index]["input_ids"]) for index in indices)
        session_tokens += input_tokens
        session_target_tokens += total_tokens
        session_steps += 1
        session_elapsed = time.perf_counter() - session_start
        step_tps = input_tokens / max(step_seconds, 1e-4)
        avg_tps = session_tokens / max(session_elapsed, 1e-4)
        avg_step_sec = session_elapsed / session_steps
        remaining_steps = total_steps - step
        eta_sec = remaining_steps * avg_step_sec
        h = int(eta_sec // 3600)
        m = int((eta_sec % 3600) // 60)
        eta_str = f"{h:02d}h {m:02d}m" if h > 0 else f"{m:02d}m {int(eta_sec % 60):02d}s"
        peak_bytes = torch.cuda.max_memory_allocated() if args.device == "cuda" else 0
        vram_gb = peak_bytes / (1024 ** 3)
        current_lr = scheduler.get_last_lr()[0]
        pct = (step / total_steps) * 100

        if use_pretty_log:
            print(f"[{step:03d}/{total_steps} | {pct:4.1f}%] ep {epoch+1}/{args.epochs} | loss: {training_loss:.4f} | "
                  f"{step_tps:5.1f} t/s (avg {avg_tps:5.1f}) | {step_seconds:4.1f}s/step | "
                  f"ETA: {eta_str} | VRAM: {vram_gb:.2f}G | lr: {current_lr:.2e}", flush=True)
        else:
            print(json.dumps({"step": step, "epoch": epoch+1, "position": position, "loss": training_loss,
                              "target_tokens": total_tokens, "input_tokens": input_tokens,
                              "tps": round(step_tps, 1), "avg_tps": round(avg_tps, 1),
                              "lr": current_lr, "seconds": round(step_seconds, 3), "eta": eta_str,
                              "peak_allocated_bytes": peak_bytes}), flush=True)
        if args.eval_every and step % args.eval_every == 0:
            evaluate()
            last_evaluated = step
        if args.save_every and step % args.save_every == 0:
            save()
            last_saved = step
    if last_evaluated != step:
        evaluate()
    if last_saved != step:
        save()


if __name__ == "__main__":
    main()
