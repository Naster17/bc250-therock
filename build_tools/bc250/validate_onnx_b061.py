#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Owned modern ONNX exports, CPU runtimes/references and explicit HIP interoperability."""

import argparse
import copy
import importlib.metadata as metadata
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group", choices=("audit", "export", "order"), required=True)
    args = parser.parse_args()
    torch_preimported = "torch" in sys.modules
    # Initialize an ORT session before the first owned Torch GPU workload.
    import numpy as np
    import onnxruntime as ort
    import onnx

    if args.group == "order":
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node("Identity", ["x"], ["y"])],
            "owned-identity",
            [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2, 3])],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2, 3])],
        )
        model = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 18)], ir_version=10
        )
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        early = ort.InferenceSession(
            model.SerializeToString(),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        early.disable_fallback()
        np.testing.assert_array_equal(
            early.run(None, {"x": np.ones((2, 3), np.float32)})[0], 1
        )
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(6101)
    assert torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1013:")
    report = dict(
        status="running",
        group=args.group,
        torch_preimported_at_script_entry=torch_preimported,
        checks=[],
        providers=ort.get_available_providers(),
    )

    def passed(name, **details):
        report["checks"].append(dict(name=name, **details))
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    try:
        if args.group == "audit":
            pins = json.loads(
                Path(__file__).with_name("onnx_b061_packages.json").read_text()
            )
            report["versions"] = {name: metadata.version(name) for name in pins}
            assert all(report["versions"][n] == p["version"] for n, p in pins.items())
            assert (
                metadata.version("numpy") == "2.5.1"
                and metadata.version("sympy") == "1.14.0"
            )
            assert (
                metadata.version("typing_extensions") == "4.16.0"
                and metadata.version("packaging") == "26.2"
            )
            assert torch.__version__ == "2.13.0a0+gfx1013"
            passed("pinned-new-packages-existing-numerical-dependencies-preserved")
            assert "CPUExecutionProvider" in ort.get_available_providers()
            assert not {
                "CUDAExecutionProvider",
                "ROCMExecutionProvider",
                "MIGraphXExecutionProvider",
            } & set(ort.get_available_providers())
            passed("cpu-provider-present-no-amd-or-cuda-provider")
            import onnxscript._framework_apis.torch_2_11
            import bc250_quant
            import bc250_fft
            import bc250_sparse

            assert callable(onnxscript._framework_apis.torch_2_11.get_torchlib_ops)
            assert (
                bc250_quant.__version__
                == bc250_fft.__version__
                == bc250_sparse.__version__
                == "0.1.0"
            )
            passed("pinned-exporter-api-and-retained-explicit-helpers")
            try:
                ort.OrtValue.ortvalue_from_numpy(np.ones(2, np.float32), "cuda", 0)
            except RuntimeError as error:
                passed("native-ort-gpu-allocation-refuses-cpu-build", error=str(error))
            else:
                raise AssertionError(
                    "CPU runtime unexpectedly allocated a GPU OrtValue"
                )
        elif args.group == "order":
            x = (
                torch.arange(21, device="cuda").float().reshape(3, 7) / 64
            ).requires_grad_()
            w = torch.arange(35, device="cuda").float().reshape(7, 5) / 128
            y = x @ w
            y.sum().backward()
            torch.testing.assert_close(
                y.cpu(),
                (x.detach().cpu().double() @ w.cpu().double()).float(),
                rtol=1e-5,
                atol=2e-6,
            )
            torch.testing.assert_close(
                x.grad.cpu(), w.cpu().sum(1).expand(3, 7), rtol=1e-6, atol=1e-6
            )
            passed("ort-first-initialization-native-gpu-gemm-first-gradient")
            t = torch.arange(32, device="cuda").float() / 32
            torch.testing.assert_close(
                torch.fft.rfft(t).cpu(),
                torch.fft.rfft(t.cpu().double()).to(torch.complex64),
                rtol=1e-5,
                atol=3e-6,
            )
            passed("ort-first-initialization-native-gpu-fft")
        else:
            from onnx.reference import ReferenceEvaluator

            class Dense(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.first = torch.nn.Linear(8, 12)
                    self.last = torch.nn.Linear(12, 4)

                def forward(self, x):
                    return self.last(torch.relu(self.first(x)))

            class CNN(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv = torch.nn.Conv2d(2, 4, 3, padding=1)
                    self.head = torch.nn.Linear(4, 3)

                def forward(self, x):
                    return self.head(torch.relu(self.conv(x)).mean(dim=(-1, -2)))

            class Gated(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.values = torch.nn.Linear(8, 8)
                    self.gates = torch.nn.Linear(8, 8)
                    self.norm = torch.nn.LayerNorm(8)
                    self.head = torch.nn.Linear(8, 4)

                def forward(self, x):
                    return self.head(
                        self.norm(self.values(x) * torch.sigmoid(self.gates(x)))
                    ).mean(dim=1)

            for family, constructor, tail in (
                ("dense", Dense, (8,)),
                ("cnn", CNN, (2, 8, 8)),
                ("gated", Gated, (5, 8)),
            ):
                original = constructor().eval()
                reference = copy.deepcopy(original).double()
                for device in ("cpu", "cuda"):
                    label = family + "-" + device
                    folder = args.output.parent / label
                    folder.mkdir()
                    model = copy.deepcopy(original).to(device)
                    sample = torch.randn((3, *tail), device=device)
                    program = torch.onnx.export(
                        model,
                        (sample,),
                        dynamo=True,
                        opset_version=18,
                        dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=4)},),
                        input_names=["x"],
                        output_names=["y"],
                        report=True,
                        artifacts_dir=str(folder),
                        external_data=False,
                    )
                    file = folder / "model.onnx"
                    program.save(file, external_data=False)
                    graph = onnx.load(file)
                    onnx.checker.check_model(graph, full_check=True)
                    assert graph.graph.input[0].type.tensor_type.shape.dim[0].dim_param
                    assert all(n.domain in ("", "ai.onnx") for n in graph.graph.node)
                    assert not any(p.grad is not None for p in model.parameters())
                    passed(
                        label + "-modern-dynamo-checked-symbolic-batch-export",
                        ops=[n.op_type for n in graph.graph.node],
                        ir=graph.ir_version,
                        opsets={x.domain: x.version for x in graph.opset_import},
                    )
                    options = ort.SessionOptions()
                    options.intra_op_num_threads = options.inter_op_num_threads = 1
                    options.enable_profiling = True
                    options.profile_file_prefix = str(folder / "ort-profile")
                    session = ort.InferenceSession(
                        str(file),
                        sess_options=options,
                        providers=["CPUExecutionProvider"],
                    )
                    session.disable_fallback()
                    assert session.get_providers() == ["CPUExecutionProvider"]
                    evaluator = ReferenceEvaluator(graph)
                    for batch in (1, 3, 4):
                        data = (
                            np.random.default_rng(6100 + batch).normal(
                                size=(batch, *tail)
                            )
                            / 4
                        ).astype(np.float32)
                        actual = session.run(["y"], {"x": data})[0]
                        with torch.no_grad():
                            gold = (
                                reference(torch.from_numpy(data).double())
                                .float()
                                .numpy()
                            )
                            eager = (
                                model(torch.from_numpy(data).to(device)).cpu().numpy()
                            )
                        np.testing.assert_allclose(actual, gold, rtol=3e-4, atol=2e-5)
                        np.testing.assert_allclose(actual, eager, rtol=3e-4, atol=2e-5)
                        np.testing.assert_allclose(
                            actual,
                            evaluator.run(["y"], {"x": data})[0],
                            rtol=3e-4,
                            atol=2e-5,
                        )
                        # Explicit host transfer; ORT stays CPU. This derivative
                        # belongs to the downstream Torch GPU operation only.
                        value = torch.from_numpy(actual.copy()).cuda().requires_grad_()
                        factors = (
                            torch.arange(value.numel(), device="cuda")
                            .reshape(value.shape)
                            .float()
                            / 32
                        )
                        (value.square() * factors).sum().backward()
                        expected = (
                            2
                            * actual.astype(np.float64)
                            * factors.cpu().double().numpy()
                        ).astype(np.float32)
                        np.testing.assert_allclose(
                            value.grad.cpu().numpy(), expected, rtol=2e-6, atol=1e-6
                        )
                        passed(
                            label
                            + "-batch-"
                            + str(batch)
                            + "-cpu-double-onnx-numpy-hip-reference-interoperability"
                        )
                    trace = Path(session.end_profiling())
                    nodes = [
                        e
                        for e in json.loads(trace.read_text())
                        if e.get("cat") == "Node" and e.get("args", {}).get("provider")
                    ]
                    assert nodes and all(
                        e["args"]["provider"] == "CPUExecutionProvider" for e in nodes
                    )
                    passed(
                        label + "-actual-onnx-node-profile-cpu-only", nodes=len(nodes)
                    )
            for name, feed in (
                ("dtype", {"x": data.astype(np.float64)}),
                ("shape", {"x": np.zeros((2, 5, 9), np.float32)}),
                ("missing-input", {}),
            ):
                try:
                    session.run(["y"], feed)
                except (
                    ValueError,
                    ort.capi.onnxruntime_pybind11_state.InvalidArgument,
                ) as error:
                    passed("native-input-refusal-" + name, error=str(error))
                else:
                    raise AssertionError("Missing runtime input refusal: " + name)
        torch.cuda.synchronize()
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
