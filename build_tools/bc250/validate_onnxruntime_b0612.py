#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Execute fresh Torch exports via the public ORT GPU route without CPU assignment."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import numpy as np
    import onnxruntime as ort
    import torch

    root = Path(os.environ["BC250_ORT_PACKAGE"])
    assert ort.__version__ == "1.31.0+gfx1013.b0611" and Path(
        ort.__file__
    ).is_relative_to(root)
    assert torch.cuda.get_device_properties(0).gcnArchName == "gfx1013:xnack-"
    baseline = json.loads((args.fixtures / "manifest.json").read_text())
    assert baseline["status"] == "passed" and len(baseline["models"]) == 6
    for name, digest in baseline["files_sha256"].items():
        assert (
            hashlib.sha256((args.fixtures / name).read_bytes()).hexdigest() == digest
        ), name
    report = dict(
        status="running",
        checks=[],
        models={},
        profiles=[],
        ort_path=ort.__file__,
        tolerance=baseline["tolerance"],
    )

    def passed(name):
        report["checks"].append(name)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    try:
        for name, spec in baseline["models"].items():
            folder = args.fixtures / name
            so = ort.SessionOptions()
            so.intra_op_num_threads = so.inter_op_num_threads = 1
            so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            so.enable_profiling = True
            so.profile_file_prefix = str(args.output.parent / ("gpu-profile-" + name))
            options = {
                k: "0"
                for k in (
                    "migraphx_fp16_enable",
                    "migraphx_bf16_enable",
                    "migraphx_fp8_enable",
                    "migraphx_int8_enable",
                    "migraphx_exhaustive_tune",
                )
            }
            options.update(device_id="0", migraphx_mem_limit=str(512 * 1024**2))
            session = ort.InferenceSession(
                str(folder / "model.onnx"),
                sess_options=so,
                providers=[("MIGraphXExecutionProvider", options)],
            )
            session.disable_fallback()
            assert session.get_providers()[0] == "MIGraphXExecutionProvider"
            assert (
                session.get_inputs()[0].shape == spec["input_shape"]
                and session.get_inputs()[0].type == "tensor(float)"
            )
            passed(name + "-export-contract-gpu-session")
            for replay in (0, 1):
                data = np.fromfile(folder / f"x-{replay}.f32", dtype="<f4").reshape(
                    spec["input_shape"]
                )
                before = data.copy()
                actual = session.run(["y"], {"x": data})[0]
                assert (
                    actual.dtype == np.float32
                    and list(actual.shape) == spec["output_shape"]
                    and np.isfinite(actual).all()
                )
                gold = np.fromfile(folder / f"cpu-{replay}.f64", dtype="<f8").reshape(
                    spec["output_shape"]
                )
                np.testing.assert_allclose(
                    actual.astype(np.float64), gold, **baseline["tolerance"]
                )
                for label in ("ort", "hip", "onnx"):
                    ref = np.fromfile(
                        folder / f"{label}-{replay}.f32", dtype="<f4"
                    ).reshape(spec["output_shape"])
                    np.testing.assert_allclose(actual, ref, **baseline["tolerance"])
                np.testing.assert_array_equal(data, before)
                passed(name + "-independent-reference-" + str(replay))
            profile = Path(session.end_profiling())
            events = json.loads(profile.read_text())
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
                    path=str(profile),
                    nodes=len(nodes),
                    providers=sorted({e["args"]["provider"] for e in nodes}),
                )
            )
            report["models"][name] = spec
            passed(name + "-actual-gpu-node-assignment")
        for label, feed in [
            ("dtype", {"x": data.astype(np.float64)}),
            ("shape", {"x": np.zeros((3, 5, 9), np.float32)}),
            ("missing", {}),
        ]:
            try:
                session.run(["y"], feed)
            except (
                ValueError,
                RuntimeError,
                ort.capi.onnxruntime_pybind11_state.InvalidArgument,
            ):
                pass
            else:
                raise AssertionError("Invalid input accepted: " + label)
            passed("contract-refusal-" + label)
        x = torch.arange(35, device="cuda", dtype=torch.float32).requires_grad_()
        x.square().sum().backward()
        torch.cuda.synchronize()
        np.testing.assert_array_equal(
            x.grad.cpu().numpy(), 2 * np.arange(35, dtype=np.float32)
        )
        passed("torch-native-backward-after-exported-ort-gpu")
        maps = Path("/proc/self/maps").read_text()
        report["maps"] = [
            s
            for s in maps.splitlines()
            if any(
                v in s
                for v in (
                    "onnxruntime",
                    "migraphx",
                    "libamdhip64",
                    "libMIOpen",
                    "librocblas",
                )
            )
        ]
        assert all(
            any(str(root) in s and library in s for s in report["maps"])
            for library in (
                "onnxruntime_pybind11_state",
                "libonnxruntime_providers_migraphx",
            )
        )
        passed("public-package-and-gpu-library-origins")
        cpu_so = ort.SessionOptions()
        cpu_so.intra_op_num_threads = cpu_so.inter_op_num_threads = 1
        cpu = ort.InferenceSession(
            str(folder / "model.onnx"),
            sess_options=cpu_so,
            providers=["CPUExecutionProvider"],
        )
        cpu.disable_fallback()
        np.testing.assert_allclose(
            cpu.run(["y"], {"x": data})[0], gold, **baseline["tolerance"]
        )
        passed("explicit-cpu-after-exported-gpu")
        assert len(report["checks"]) == 30
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
