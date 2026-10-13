# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Short standalone release smoke: GPU, Inductor, frozen PEFT and checkpoint IO."""

import argparse
import importlib.metadata
import json
from pathlib import Path

import torch
import triton
import transformers
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM
from torch.nn.attention import SDPBackend, sdpa_kernel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(1704)
    properties = torch.cuda.get_device_properties(0)
    assert properties.gcnArchName.split(":")[0] == "gfx1013"
    assert Path(torch.__file__).is_relative_to(Path("/workspace/pytorch-work"))
    assert "site-packages/triton/" in triton.__file__
    assert Path(transformers.__file__).is_relative_to(
        Path("/workspace/release-consumers")
    )
    checks = []
    values = torch.linspace(-1, 1, 65539)
    compiled = torch.compile(lambda x: torch.sin(x) + x.square(), fullgraph=True)
    torch.testing.assert_close(
        compiled(values.cuda()).cpu(),
        torch.sin(values) + values.square(),
        rtol=1e-5,
        atol=1e-5,
    )
    checks.append("inductor_gpu_cpu_reference_without_development_mounts")
    configuration = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        use_cache=False,
        attention_dropout=0,
    )
    configuration._attn_implementation = "sdpa"
    model = get_peft_model(
        LlamaForCausalLM(configuration),
        LoraConfig(
            task_type="CAUSAL_LM",
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0,
        ),
    ).cuda()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.01
    )
    tokens = torch.arange(16, device="cuda").unsqueeze(0)
    losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with sdpa_kernel(SDPBackend.MATH):
            loss = model(input_ids=tokens, labels=tokens).loss
        assert torch.isfinite(loss).item()
        loss.backward()
        assert all(
            p.grad is None or torch.isfinite(p.grad).all().item()
            for p in model.parameters()
        )
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    checks.append("frozen_transformers_peft_lora_gpu_updates")
    checkpoint = args.output.with_suffix(".pt")
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint
    )
    loaded = torch.load(checkpoint, weights_only=True, map_location="cpu")
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(loaded["model"][name], tensor.cpu(), rtol=0, atol=0)
    checks.append("host_visible_checkpoint_roundtrip")
    args.output.write_text(
        json.dumps(
            {
                "status": "passed",
                "arch": properties.gcnArchName,
                "checks": checks,
                "losses": losses,
                "torch_path": torch.__file__,
                "triton_path": triton.__file__,
                "transformers_path": transformers.__file__,
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in (
                        "torch",
                        "triton",
                        "transformers",
                        "peft",
                        "accelerate",
                    )
                },
            },
            indent=2,
        )
        + "\n"
    )
    print("STANDALONE_RELEASE_SMOKE_PASS", flush=True)


if __name__ == "__main__":
    main()
