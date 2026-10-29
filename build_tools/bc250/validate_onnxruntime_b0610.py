#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check real isolated ORT MIGraphX assignment and references; reject silent fallback."""
import argparse
import gc
import importlib
import json
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--order", choices=("ort", "torch", "migraphx"), required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert not {"torch", "onnxruntime", "migraphx"} & sys.modules.keys()
    sys.path.insert(0, str(args.candidate))
    orders = {
        "ort": ("onnxruntime", "migraphx", "torch"),
        "torch": ("torch", "onnxruntime", "migraphx"),
        "migraphx": ("migraphx", "onnxruntime", "torch"),
    }
    modules = {name: importlib.import_module(name) for name in orders[args.order]}
    ort, torch, mx = (modules[n] for n in ("onnxruntime", "torch", "migraphx"))
    import numpy as np

    sdk = Path(os.environ["BC250_MIGRAPHX_SDK"])
    fixtures = sdk / "share/bc250-sdk/fixtures"
    baseline = json.loads((sdk / "share/bc250-sdk/fixtures.json").read_text())
    report = dict(
        status="running",
        checks=[],
        order=list(orders[args.order]),
        candidate=str(args.candidate),
        ort_path=ort.__file__,
        providers=ort.get_available_providers(),
        profiles=[],
    )

    def passed(name):
        report["checks"].append(name)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    assert (
        Path(ort.__file__).is_relative_to(args.candidate)
        and ort.__version__ == "1.31.0"
    )
    assert "MIGraphXExecutionProvider" in report["providers"]
    passed("matching-isolated-ort-module-and-gpu-provider")
    assert torch.cuda.get_device_properties(0).gcnArchName == "gfx1013:xnack-"
    assert os.environ["HSA_ENABLE_SDMA"] == "0"
    passed("native-gfx1013-sdma-off")
    provider_options = {
        k: "0"
        for k in (
            "migraphx_fp16_enable",
            "migraphx_bf16_enable",
            "migraphx_fp8_enable",
            "migraphx_int8_enable",
            "migraphx_exhaustive_tune",
        )
    }
    provider_options.update(device_id="0", migraphx_mem_limit=str(512 * 1024**2))

    def options(profile=False):
        so = ort.SessionOptions()
        so.intra_op_num_threads = so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        so.enable_profiling = profile
        so.profile_file_prefix = str(args.output.parent / "profile")
        return so

    first_model = None
    for name, spec in baseline["models"].items():
        folder = fixtures / name
        session = ort.InferenceSession(
            str(folder / "model.onnx"),
            sess_options=options(True),
            providers=[("MIGraphXExecutionProvider", provider_options)],
        )
        session.disable_fallback()
        assert session.get_providers()[0] == "MIGraphXExecutionProvider"
        passed(name + "-explicit-gpu-no-cpu-fallback-session")
        last = None
        for replay in (0, 1):
            inputs = {
                key: np.fromfile(folder / f"{key}-{replay}.bin", dtype="<f4").reshape(
                    shape
                )
                for key, shape in spec["inputs"].items()
            }
            before = {k: v.copy() for k, v in inputs.items()}
            output = session.run(None, inputs)
            assert len(output) == 1
            actual = output[0]
            assert actual.dtype == np.float32 and list(actual.shape) == spec["output"]
            cpu = np.fromfile(folder / f"cpu-{replay}.f64", dtype="<f8").reshape(
                spec["output"]
            )
            old = np.fromfile(folder / f"ort-{replay}.f32", dtype="<f4").reshape(
                spec["output"]
            )
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(
                actual.astype(np.float64), cpu, atol=2e-5, rtol=2e-5
            )
            np.testing.assert_allclose(actual, old, atol=2e-5, rtol=2e-5)
            for key in inputs:
                np.testing.assert_array_equal(inputs[key], before[key])
            passed(name + "-reference-changed-input-" + str(replay))
            last = inputs
        if first_model is None:
            first_model = (session, last)
        path = Path(session.end_profiling())
        events = json.loads(path.read_text())
        nodes = [
            e
            for e in events
            if e.get("cat") == "Node" and e.get("args", {}).get("provider")
        ]
        assert nodes and all(
            e["args"]["provider"] == "MIGraphXExecutionProvider" for e in nodes
        ), nodes
        report["profiles"].append(
            dict(
                model=name,
                path=str(path),
                providers=sorted({e["args"]["provider"] for e in nodes}),
                nodes=len(nodes),
            )
        )
        passed(name + "-actual-profile-gpu-node-assignment")
        del session
        gc.collect()
    session, inputs = first_model
    for name, bad in [
        ("dtype", {k: v.astype(np.float64) for k, v in inputs.items()}),
        ("shape", {k: np.concatenate([v, v], axis=0) for k, v in inputs.items()}),
        ("missing", {}),
    ]:
        try:
            session.run(None, bad)
        except (
            ValueError,
            RuntimeError,
            ort.capi.onnxruntime_pybind11_state.InvalidArgument,
        ):
            pass
        else:
            raise AssertionError("Invalid input accepted: " + name)
        passed("contract-refusal-" + name)
    folder = fixtures / "constant"
    so = ort.SessionOptions()
    so.intra_op_num_threads = so.inter_op_num_threads = 1
    cpu_session = ort.InferenceSession(
        str(folder / "model.onnx"), sess_options=so, providers=["CPUExecutionProvider"]
    )
    cpu_session.disable_fallback()
    value = np.fromfile(folder / "x-1.bin", dtype="<f4").reshape(3, 17)
    reference = np.fromfile(folder / "cpu-1.f64", dtype="<f8").reshape(3, 5)
    np.testing.assert_allclose(
        cpu_session.run(None, {"x": value})[0], reference, rtol=2e-5, atol=2e-5
    )
    passed("candidate-explicit-cpu-provider-after-gpu")
    x = torch.arange(35, device="cuda", dtype=torch.float32).requires_grad_()
    x.square().sum().backward()
    torch.cuda.synchronize()
    np.testing.assert_array_equal(
        x.grad.cpu().numpy(), 2 * np.arange(35, dtype=np.float32)
    )
    passed("torch-native-first-backward-after-candidate")
    report["maps"] = [
        s
        for s in Path("/proc/self/maps").read_text().splitlines()
        if any(
            k in s
            for k in (
                "onnxruntime",
                "migraphx",
                "protobuf",
                "absl",
                "libamdhip64",
                "libMIOpen",
                "librocblas",
            )
        )
    ]
    for library in (
        "onnxruntime_pybind11_state",
        "libonnxruntime_providers_migraphx",
        "libonnxruntime_providers_shared",
    ):
        assert any(
            library in line and str(args.candidate) in line for line in report["maps"]
        ), library
    assert any(
        "libmigraphx_onnx.so" in line and str(sdk) in line for line in report["maps"]
    )
    passed("actual-matching-candidate-and-sdk-library-origins")
    assert len(report["checks"]) == 32, len(report["checks"])
    report["status"] = "passed"
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
