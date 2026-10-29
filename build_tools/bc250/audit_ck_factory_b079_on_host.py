#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Pin existing FP16 factory, float accumulator and native gfx1013 fallback sources."""
import json
from pathlib import Path
from bc250_torch import runtime_lock
from deploy_torch_a04_on_host import atomic_json, sha
from qualify_training_n33_on_host import WS


def main():
    root = WS / "tmp/ck-plugin-b079"
    pins = json.loads((root / "audit-inputs.json").read_text())
    stage = WS / "build/math-libs/composable_kernel/stage"
    checked = {}
    with runtime_lock(Path("/opt/bc250-stack/current/.runtime.lock")):
        for item in pins:
            source = WS / "source/rocm-libraries" / item["source"]
            assert sha(source) == item["sha256"], str(source)
            checked[str(source)] = sha(source)
            if item["stage"]:
                target = stage / item["stage"]
                assert sha(target) == item["sha256"], str(target)
                checked[str(target)] = sha(target)
        instance = next(
            p for p in checked if p.endswith("device_grouped_conv_fwd_dl_instance.hpp")
        )
        text = Path(instance).read_text()
        fp16 = text.split("using device_grouped_conv2d_fwd_dl_f16_instances", 1)[
            1
        ].split("using device_grouped_conv2d_fwd_dl_f32_instances", 1)[0]
        lines = [
            line
            for line in fp16.splitlines()
            if "DeviceGroupedConvFwdDlMultipleD_" in line
        ]
        assert len(lines) == 3
        assert all(
            "F16,     F16,  DsDatatype,     F16,     F32" in line for line in lines
        )
        factory = (
            stage
            / "include/ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp"
        )
        text = factory.read_text()
        assert "#ifdef DL_KERNELS" in text and "#ifdef CK_ENABLE_FP16" in text
        config = stage / "include/ck/config.h"
        archive = stage / "lib/libdevice_conv_operations.a"
        checked[str(config)] = sha(config)
        checked[str(archive)] = sha(archive)
        atomic_json(
            root / "source-audit.json",
            dict(
                status="passed",
                source_stage_sha256=checked,
                FP16_input_output=True,
                FP32_accumulator_source_rows=3,
                required_factory_macro="DL_KERNELS",
                layouts=["NHWGC", "GNHWC"],
                arithmetic_scope="gfx1013 avoids CK_USE_AMD_V_DOT2_F32_F16; scalar half-to-float inner products and FP32 accumulator. Device macro proof is in the numerical probe.",
                runtime_changed=False,
                host105_contacted=False,
            ),
        )
    print("CK_PLUGIN_B079_SOURCE_AUDIT_PASS", flush=True)


if __name__ == "__main__":
    main()
