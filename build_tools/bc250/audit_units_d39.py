#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""D39 CPU-reference clipping audit, with the D36 legacy control preserved."""
import math
import torch
import distributed_d35_growth as base
from audit_units_d36 import Auditor as LegacyAuditor, tensor_sha


class Auditor(LegacyAuditor):
    def __init__(self, kind, scratch):
        if kind not in {"legacy", "reference"}:
            raise ValueError("Unknown audit selection")
        self.selection = kind
        super().__init__("unit", scratch)

    def qualify(
        self, model, opt, rank, world, traffic, monitor, manifest_path, manifest
    ):
        if self.selection == "legacy":
            result = super().qualify(
                model, opt, rank, world, traffic, monitor, manifest_path, manifest
            )
            result["clipping_audit"] = "legacy-actual-gradient-spool"
            return result
        assert world == 2
        monitor.phase = "D39-GPU-backward-and-audit"
        torch.cuda.reset_peak_memory_stats()
        directory = manifest_path.parent
        entries = {entry["name"]: entry for entry in manifest["tensors"]}
        assert not self.scratch.exists()
        raw_digests = {}
        opt.zero_grad(set_to_none=True)
        actual_loss = base.fixture.gpu_backward(model, 0, rank, world, traffic)
        base.dist.all_reduce(actual_loss)
        actual_loss /= world
        torch.testing.assert_close(
            actual_loss.cpu().double(),
            torch.tensor(manifest["CPU_FP64_loss"], dtype=torch.float64),
            atol=3e-6,
            rtol=5e-5,
        )
        squares = {}
        max_error = 0.0
        for name, p in self.parameters(model, entries, "raw-gradient"):
            entry = entries[name]
            assert p.is_cuda and p.dtype == p.grad.dtype == torch.float32
            initial = base.tensor_blob(
                directory, entry["initial"], entry["shape"], torch.float32
            )
            assert torch.equal(p.detach().cpu(), initial)
            expected = base.tensor_blob(
                directory, entry["gradient"], entry["shape"], torch.float64
            )
            raw = p.grad.detach().cpu().contiguous()
            actual = raw.double()
            torch.testing.assert_close(actual, expected, atol=3e-6, rtol=5e-4)
            max_error = max(max_error, float((actual - expected).abs().max()))
            squares[name] = float(actual.square().sum())
            raw_digests[name] = tensor_sha(raw)
            del initial, expected, raw, actual
        del p
        # Preserve original canonical CPU summation order despite unit traversal.
        wanted_norm = math.sqrt(sum(squares[name] for name in entries))
        norm = base.clip(model)
        torch.testing.assert_close(
            norm.cpu().double(),
            torch.tensor(wanted_norm, dtype=torch.float64),
            atol=1e-7,
            rtol=1e-5,
        )
        coefficient = min(1.0, 0.5 / (wanted_norm + 1e-6))
        clipped_digests = {}
        maximum_absolute_clip_error = 0.0
        for name, p in self.parameters(model, entries, "clipped-gradient"):
            raw = base.tensor_blob(
                directory,
                entries[name]["gradient"],
                entries[name]["shape"],
                torch.float64,
            )
            torch.testing.assert_close(
                p.grad.detach().cpu().double(), raw * coefficient, atol=1e-8, rtol=5e-5
            )
            maximum_absolute_clip_error = max(
                maximum_absolute_clip_error,
                float((p.grad.detach().cpu().double() - raw * coefficient).abs().max()),
            )
            clipped_digests[name] = tensor_sha(p.grad)
            del raw
        del p
        assert not self.scratch.exists()
        monitor.phase = "D39-AdamW-and-ALL-state-audit"
        opt.step()
        maximum_CPU_moment_elements = 0
        updated_digests = {}
        for name, p in self.parameters(model, entries, "updated-parameter"):
            entry = entries[name]
            initial = base.tensor_blob(
                directory, entry["initial"], entry["shape"], torch.float32
            )
            ref = torch.nn.Parameter(initial.double())
            ref.grad = p.grad.detach().cpu().double()
            refopt = base.fixture.optimizer(torch.nn.ParameterList([ref]))
            refopt.step()
            maximum_CPU_moment_elements = max(
                maximum_CPU_moment_elements, 2 * ref.numel()
            )
            torch.testing.assert_close(
                p.detach().cpu().double(), ref.detach(), atol=4e-6, rtol=5e-5
            )
            updated_digests[name] = tensor_sha(p)
            del initial, refopt, ref
        del p
        local_elements = 0
        state_digests = {}
        for name, p in base.fixture.named(model).items():
            if p.numel() == 0:
                continue
            assert p.grad is not None
            state = opt.state[p]
            grad = p.grad.detach().cpu().double()
            for key, expected_moment, atol, rtol in [
                ("exp_avg", grad * 0.1, 3e-7, 5e-4),
                ("exp_avg_sq", grad.square() * 0.001, 3e-9, 1e-3),
            ]:
                torch.testing.assert_close(
                    state[key].cpu().double(), expected_moment, atol=atol, rtol=rtol
                )
            assert int(state["step"]) == 1
            state_digests[name] = {
                key: tensor_sha(state[key]) for key in ["exp_avg", "exp_avg_sq", "step"]
            }
            local_elements += p.numel()
        assert local_elements == base.PARAMETERS // world
        return dict(
            loss=float(actual_loss),
            CPU_FP64_loss=manifest["CPU_FP64_loss"],
            gradient_parameters=len(entries),
            gradient_elements=base.PARAMETERS,
            maximum_absolute_gradient_error=max_error,
            root_norm=float(norm),
            CPU_root_norm=wanted_norm,
            clip_coefficient=coefficient,
            ALL_AdamW_parameters_moments_counters=True,
            local_moment_elements_each=local_elements,
            maximum_CPU_moment_elements=maximum_CPU_moment_elements,
            actual_raw_spool_bytes=0,
            actual_raw_spool_removed=True,
            actual_raw_spool_sha256=raw_digests,
            clipping_audit="independent-CPU-FP64-reference",
            maximum_absolute_CPU_reference_clip_error=maximum_absolute_clip_error,
            clipping_atol=1e-8,
            clipping_rtol=5e-5,
            full_CPU_optimizer_or_export=False,
            CPU_oracle_executed_in_GPU_worker=False,
            clipped_gradient_sha256=clipped_digests,
            updated_parameter_sha256=updated_digests,
            local_AdamW_sha256=state_digests,
            audit=dict(
                kind=self.kind,
                ownership=self.ownership,
                events=self.events,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                whole_model_summons_forbidden=self.kind == "unit",
            ),
        )
