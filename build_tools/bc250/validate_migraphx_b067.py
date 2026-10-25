#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Isolated Python ONNX/GPU and serialization references in fresh import orders."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

ORDERS = {
    "migraphx": ("migraphx", "onnxruntime", "torch"),
    "ort": ("onnxruntime", "torch", "migraphx"),
    "torch": ("torch", "migraphx", "onnxruntime"),
}


def worker(args, storage, baseline):
    assert not {"torch", "onnxruntime", "migraphx"} & sys.modules.keys()
    modules = {name: importlib.import_module(name) for name in ORDERS[args.order]}
    mx, torch, ort = (modules[n] for n in ("migraphx", "torch", "onnxruntime"))
    import numpy as np

    report = dict(
        status="running",
        checks=[],
        order=list(ORDERS[args.order]),
        binding=str(mx.__file__),
        versions={n: m.__version__ for n, m in modules.items()},
        inference_only=True,
    )

    def passed(name):
        report["checks"].append(name)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def compare(program, inputs, folder, replay, shape, explicit=False):
        before = {n: v.copy() for n, v in inputs.items()}
        params = (
            {n: mx.to_gpu(mx.argument(v)) for n, v in inputs.items()}
            if explicit
            else inputs
        )
        if explicit:
            for name, output_shape in program.get_parameter_shapes().items():
                if name not in params:
                    assert name.startswith("main:#output_"), name
                    params[name] = mx.allocate_gpu(output_shape)
            assert set(params) == set(program.get_parameter_shapes())
        outputs = program.run(params)
        assert len(outputs) == 1
        result = mx.from_gpu(outputs[0]) if explicit else outputs[0]
        actual = np.array(result, copy=True)
        assert actual.dtype == np.float32 and list(actual.shape) == shape
        cpu = np.fromfile(folder / f"cpu-{replay}.f64", dtype="<f8").reshape(shape)
        ref = np.fromfile(folder / f"ort-{replay}.f32", dtype="<f4").reshape(shape)
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual.astype(np.float64), cpu, atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(actual, ref, atol=2e-5, rtol=2e-5)
        for name, value in inputs.items():
            np.testing.assert_array_equal(value, before[name])
        return actual

    try:
        assert Path(mx.__file__).parent == storage / "lib"
        assert torch.cuda.get_device_properties(0).gcnArchName == "gfx1013:xnack-"
        assert not {
            "MIGraphXExecutionProvider",
            "ROCMExecutionProvider",
            "CUDAExecutionProvider",
        } & set(ort.get_available_providers())
        passed("isolated-binding-native-device-cpu-only-ort-provider")
        for name, spec in baseline["models"].items():
            folder = storage / "share/bc250-sdk/fixtures" / name
            payload = (folder / "model.onnx").read_bytes()
            p = mx.parse_onnx(str(folder / "model.onnx"))
            q = mx.parse_onnx_buffer(payload)
            assert p == q and not p.is_compiled()
            params = p.get_parameter_shapes()
            assert set(params) == set(spec["inputs"])
            for key, dims in spec["inputs"].items():
                assert (
                    params[key].lens() == dims
                    and params[key].type_string() == "float_type"
                )
            out = p.get_output_shapes()
            assert (
                len(out) == 1
                and out[0].lens() == spec["output"]
                and out[0].type_string() == "float_type"
            )
            passed(name + "-file-buffer-shape-dtype")
            for fmt in ("msgpack", "json"):
                path = args.output.parent / (name + "." + fmt)
                mx.save(p, str(path), format=fmt)
                loaded = mx.load(str(path), format=fmt)
                assert loaded == p and not loaded.is_compiled()
            assert mx.load_buffer(mx.save_buffer(p)) == p
            passed(name + "-uncompiled-file-json-msgpack-buffer-roundtrip")
            # Compile the buffer-parsed graph, separately exercising the Python route.
            q.compile(
                mx.get_target("gpu"),
                offload_copy=True,
                fast_math=False,
                exhaustive_tune=False,
            )
            ir = str(q)
            assert "gpu::" in ir and q.is_compiled()
            expected = (
                "gpu::convolution"
                if name == "conv"
                else (
                    "gpu::gemm"
                    if name in ("dense", "transpose", "constant")
                    else "gpu::code_object"
                )
            )
            assert expected in ir, ir
            if name == "constant":
                assert "gpu::literal" in ir
            passed(name + "-native-gpu-lowering")
            for replay in (0, 1):
                inputs = {
                    key: np.fromfile(
                        folder / f"{key}-{replay}.bin", dtype="<f4"
                    ).reshape(dims)
                    for key, dims in spec["inputs"].items()
                }
                actual = compare(q, inputs, folder, replay, spec["output"])
                passed(name + f"-gpu-double-ort-input-guards-{replay}")
            # The final changed inputs are also evaluated after compiled serialization.
            blob = mx.save_buffer(q)
            restored = mx.load_buffer(blob)
            assert restored.is_compiled()
            compare(restored, inputs, folder, 1, spec["output"])
            path = args.output.parent / (name + "-compiled.mxr")
            mx.save(q, str(path))
            restored_file = mx.load(str(path))
            assert restored_file.is_compiled()
            compare(restored_file, inputs, folder, 1, spec["output"])
            passed(name + "-compiled-file-buffer-replay")
            # Explicit same-device HIP transfer path with automatic copies disabled.
            p.compile(
                mx.get_target("gpu"),
                offload_copy=False,
                fast_math=False,
                exhaustive_tune=False,
            )
            compare(p, inputs, folder, 1, spec["output"], explicit=True)
            passed(name + "-explicit-hip-copy-replay")
        for name, message in (
            ("unknown", "Unknown operator: BC250Unknown"),
            ("corrupt", "Failed reading onnx file"),
        ):
            path = storage / "share/bc250-sdk/fixtures" / (name + ".onnx")
            for route in ("file", "buffer"):
                try:
                    (
                        mx.parse_onnx(str(path))
                        if route == "file"
                        else mx.parse_onnx_buffer(path.read_bytes())
                    )
                except RuntimeError as error:
                    assert message in str(error), str(error)
                else:
                    raise AssertionError(
                        "Malformed/unknown graph unexpectedly accepted"
                    )
                passed(name + "-" + route + "-refusal")
        # Run all consumers after MIGraphX to catch import/symbol or runtime damage.
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        folder = storage.parent / "b065/models-r2/constant"
        session = ort.InferenceSession(
            str(folder / "model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        session.disable_fallback()
        value = np.fromfile(folder / "x-1.bin", dtype="<f4").reshape(3, 17)
        reference = np.fromfile(folder / "ort-1.f32", dtype="<f4").reshape(3, 5)
        np.testing.assert_allclose(
            session.run(None, {"x": value})[0], reference, atol=2e-5, rtol=2e-5
        )
        assert session.get_providers() == ["CPUExecutionProvider"]
        passed("cpu-ort-session-after-migraphx")
        x = (
            torch.arange(35, device="cuda", dtype=torch.float32)
            .reshape(5, 7)
            .requires_grad_()
        )
        (x.square().sum() / 16).backward()
        torch.cuda.synchronize()
        np.testing.assert_array_equal(
            x.grad.cpu().numpy(), 2 * np.arange(35, dtype=np.float32).reshape(5, 7) / 16
        )
        passed("torch-hip-backward-after-migraphx")
        report["maps"] = [
            line
            for line in Path("/proc/self/maps").read_text().splitlines()
            if any(
                k in line
                for k in (
                    "migraphx",
                    "protobuf",
                    "absl",
                    "onnxruntime",
                    "libamdhip64",
                    "libsqlite3",
                    "libMIOpen",
                    "librocblas",
                    "libhiprtc",
                    "libhsa-runtime64",
                )
            )
        ]
        maps = "\n".join(report["maps"])
        for library in ("libmigraphx.so", "libmigraphx_gpu.so", "libmigraphx_onnx.so"):
            assert any(
                str(storage / "lib/migraphx/lib") in line and library in line
                for line in maps.splitlines()
            ), library
        assert any(
            "/workspace/build/dist/rocm/lib/rocm_sysdeps/lib/" in line
            and "libsqlite3" in line
            for line in maps.splitlines()
        )
        assert len(report["checks"]) == 49
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", choices=ORDERS, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--attempt", choices=("r1", "r2", "r3"), default="r1")
    args = parser.parse_args()
    storage = Path(os.environ["BC250_MIGRAPHX_SDK"])
    baseline = json.loads((storage / "share/bc250-sdk/fixtures.json").read_text())
    assert baseline["status"] == "passed" and len(baseline["checks"]) == 12
    for name, digest in baseline["sha256"].items():
        assert (
            hashlib.sha256(
                (storage / "share/bc250-sdk/fixtures" / name).read_bytes()
            ).hexdigest()
            == digest
        )
    assert not Path("/workspace/task/build-r1").exists()
    assert not Path(
        "/home/nik/bc250work/rocm-10.2-gfx1013/tmp/migraphx-b064/20261025-storage/mount/b066/build-r1"
    ).exists()
    stop = args.output.parent / "observer-stop"
    observed = args.output.parent / "observed-driver.json"
    with (args.output.parent / "observer.log").open("w") as stream:
        proc = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).with_name("observe_migraphx_b067.py")),
                "--parent",
                str(os.getpid()),
                "--stop",
                str(stop),
                "--output",
                str(observed),
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        try:
            worker(args, storage, baseline)
        finally:
            stop.touch()
            proc.wait(timeout=10)
    assert proc.returncode == 0
    evidence = json.loads(observed.read_text())
    assert (
        str(storage / "lib/migraphx/lib/migraphx-hiprtc-driver")
        in evidence["driver_executables"]
    ), evidence
    report = json.loads(args.output.read_text())
    report["checks"].append("relocated-isolated-hiprtc-driver-observed")
    report["driver_observation"] = evidence
    assert len(report["checks"]) == 50
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
