#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Actual PEFT LoRA fixture using the retained frozen FSDP reference harness."""
import copy
import sys
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora.layer import Linear
import distributed_d26_trainer as causal
import distributed_d27_trainer as native
from distributed_d25_trainer import main
from distributed_d10_restart import rng_state, exact
from bc250_fsdp_peft import BC250PeftFSDPTrainer, peft_fp32_policy, signature

name = "PEFT0.18.1 default rank4 causal LoRA FP32 and full frozen-base FSDP checkpoint"
frozen = True
layer = causal.layer
dataset = causal.dataset
cpu_loss = causal.cpu_loss
loss = causal.loss
collator = causal.collator
peft = True
observed_modules = causal.observed_modules
trainer_class = BC250PeftFSDPTrainer
policy = peft_fp32_policy
policy_keyword = "peft_parameter_policy"
clip_limit = 0.005
wrap_units = 19
reduction_units = 16
check_calls = 0
plain_base = None
lora_config = None


def contracts():
    result = native.contracts()
    probe = get_peft_model(causal.model(), copy.deepcopy(lora_config))
    policy = peft_fp32_policy(probe)
    assert signature(probe) == policy.signature_json
    result.append("pinned-single-default-PEFT-policy")
    target = next(m for m in probe.modules() if isinstance(m, Linear))
    target.scaling["default"] *= 2
    try:
        signature(probe)
    except ValueError:
        pass
    else:
        raise AssertionError("Changed LoRA scaling admitted")
    target.scaling["default"] /= 2
    result.append("changed-LoRA-scaling-refusal")
    with probe.disable_adapter():
        try:
            signature(probe)
        except ValueError:
            pass
        else:
            raise AssertionError("Disabled adapter admitted")
    result.append("disabled-adapter-refusal")
    probe.add_adapter("other", copy.deepcopy(lora_config))
    try:
        peft_fp32_policy(probe)
    except ValueError:
        pass
    else:
        raise AssertionError("Multiple adapters admitted")
    result.append("multiple-adapter-refusal")
    return result


def model():
    global plain_base, lora_config
    base = causal.model()
    plain_base = copy.deepcopy(base)
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        target_modules=["c_attn", "c_proj", "c_fc"],
        fan_in_fan_out=True,
        init_lora_weights=True,
    )
    result = get_peft_model(base, lora_config)
    assert sum(p.numel() for p in result.parameters()) == 24352
    assert sum(p.numel() for p in result.parameters() if p.requires_grad) == 3072
    return result


def check(value):
    global check_calls
    weight = value.get_input_embeddings().weight
    assert weight is value.get_output_embeddings().weight and not weight.requires_grad
    aliases = {
        n.replace("_fsdp_wrapped_module.", "")
        for n, p in value.named_parameters(remove_duplicate=False)
        if p is weight
    }
    assert aliases == {
        "base_model.model.transformer.wte.weight",
        "base_model.model.lm_head.weight",
    }, aliases
    assert sum(isinstance(m, Linear) for m in value.modules()) == 8
    check_calls += 1


def check_frozen(value, initial, *, full_gradient_views=False):
    return native.check_frozen(value, initial, full_gradient_views=full_gradient_views)


def export_audit(state, folder, data):
    before = rng_state()
    with torch.random.fork_rng(devices=[]):
        base = copy.deepcopy(plain_base)
        source = {
            n.removeprefix("base_model.model.").replace(".base_layer.", "."): p
            for n, p in state.items()
            if ".lora_" not in n
        }
        assert set(source) == set(dict(base.named_parameters()))
        with torch.no_grad():
            for n, p in base.named_parameters():
                p.copy_(source[n])
        expected = get_peft_model(copy.deepcopy(base), copy.deepcopy(lora_config))
        loaded = PeftModel.from_pretrained(
            copy.deepcopy(base), str(folder), is_trainable=True, local_files_only=True
        )
        assert (
            set(state)
            == set(dict(expected.named_parameters()))
            == set(dict(loaded.named_parameters()))
        )
        with torch.no_grad():
            for n, p in expected.named_parameters():
                p.copy_(state[n])
            for n, p in loaded.named_parameters():
                assert torch.equal(p, state[n]), n
            expected.eval()
            loaded.eval()
            left = expected(input_ids=data.ids[:3], attention_mask=data.mask[:3]).logits
            right = loaded(input_ids=data.ids[:3], attention_mask=data.mask[:3]).logits
            assert torch.equal(left, right)
    exact(rng_state(), before)
    assert (folder / "adapter_model.safetensors").is_file() and (
        folder / "adapter_config.json"
    ).is_file()
    return dict(
        parameters_exact=True,
        logits_exact=True,
        rng_unchanged=True,
        adapter_bytes=(folder / "adapter_model.safetensors").stat().st_size,
    )


def mutate(value):
    target = next(m for m in value.modules() if isinstance(m, Linear))
    target.scaling["default"] *= 2


if __name__ == "__main__":
    if "--retained-control" in sys.argv:
        sys.argv.remove("--retained-control")
        main(fixture=native)
    else:
        main(fixture=sys.modules[__name__])
