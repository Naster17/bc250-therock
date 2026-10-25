#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Create owned ONNX fixtures with CPU-double and explicit CPU ORT references."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import helper as h, TensorProto as tp, numpy_helper as nh

    args.directory.mkdir()
    report = dict(
        status="running",
        checks=[],
        models={},
        versions=dict(
            onnx=onnx.__version__, onnxruntime=ort.__version__, numpy=np.__version__
        ),
    )

    def save_case(name, nodes, shapes, output_shape, oracle, initializers=()):
        folder = args.directory / name
        folder.mkdir()
        graph = h.make_graph(
            nodes,
            name,
            [h.make_tensor_value_info(k, tp.FLOAT, s) for k, s in shapes.items()],
            [h.make_tensor_value_info("y", tp.FLOAT, output_shape)],
            initializer=list(initializers),
        )
        model = h.make_model(
            graph,
            opset_imports=[h.make_opsetid("", 18)],
            ir_version=10,
            producer_name="bc250-owned-b065",
        )
        onnx.checker.check_model(model)
        onnx.save(model, folder / "model.onnx")
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            model.SerializeToString(),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        session.disable_fallback()
        assert session.get_providers() == ["CPUExecutionProvider"]
        for replay in (0, 1):
            inputs = {
                key: (
                    (np.arange(np.prod(shape)).reshape(shape) + replay + index * 3) % 23
                    - 11
                ).astype(np.float32)
                / 16
                for index, (key, shape) in enumerate(shapes.items())
            }
            reference = oracle(
                {key: value.astype(np.float64) for key, value in inputs.items()}
            )
            actual = session.run(None, inputs)[0]
            assert (
                list(actual.shape) == output_shape and reference.shape == actual.shape
            )
            np.testing.assert_allclose(
                actual.astype(np.float64), reference, atol=2e-5, rtol=2e-5
            )
            for key, value in inputs.items():
                value.astype("<f4").tofile(folder / f"{key}-{replay}.bin")
            reference.astype("<f8").tofile(folder / f"cpu-{replay}.f64")
            actual.astype("<f4").tofile(folder / f"ort-{replay}.f32")
            report["checks"].append(f"{name}-cpu-ort-reference-{replay}")
        report["models"][name] = dict(
            inputs=shapes,
            output=output_shape,
            ir_version=10,
            opset=18,
            providers=session.get_providers(),
        )

    try:
        save_case(
            "pointwise",
            [h.make_node("Add", ["x", "b"], ["s"]), h.make_node("Relu", ["s"], ["y"])],
            dict(x=[3, 17], b=[3, 1]),
            [3, 17],
            lambda d: np.maximum(d["x"] + d["b"], 0),
        )
        axes = nh.from_array(np.array([1], np.int64), name="axes")
        save_case(
            "reduce",
            [h.make_node("ReduceSum", ["x", "axes"], ["y"], keepdims=1)],
            dict(x=[3, 129]),
            [3, 1],
            lambda d: d["x"].sum(axis=1, keepdims=True),
            [axes],
        )
        save_case(
            "dense",
            [
                h.make_node("MatMul", ["x", "w"], ["m"]),
                h.make_node("Add", ["m", "b"], ["s"]),
                h.make_node("Relu", ["s"], ["y"]),
            ],
            dict(x=[3, 17], w=[17, 5], b=[5]),
            [3, 5],
            lambda d: np.maximum(d["x"] @ d["w"] + d["b"], 0),
        )
        save_case(
            "transpose",
            [
                h.make_node("Transpose", ["x"], ["t"], perm=[1, 0]),
                h.make_node("MatMul", ["t", "w"], ["y"]),
            ],
            dict(x=[17, 3], w=[17, 5]),
            [3, 5],
            lambda d: d["x"].T @ d["w"],
        )

        def convolution(d):
            out = np.zeros((2, 3, 3, 3), np.float64)
            for b in range(2):
                for o in range(3):
                    for r in range(3):
                        for c in range(3):
                            out[b, o, r, c] = np.maximum(
                                np.sum(d["x"][b, :, r : r + 3, c : c + 3] * d["w"][o])
                                + d["b"][o],
                                0,
                            )
            return out

        save_case(
            "conv",
            [
                h.make_node(
                    "Conv", ["x", "w", "b"], ["c"], pads=[0, 0, 0, 0], strides=[1, 1]
                ),
                h.make_node("Relu", ["c"], ["y"]),
            ],
            dict(x=[2, 2, 5, 5], w=[3, 2, 3, 3], b=[3]),
            [2, 3, 3, 3],
            convolution,
        )
        unknown = h.make_model(
            h.make_graph(
                [h.make_node("BC250Unknown", ["x"], ["y"])],
                "refusal",
                [h.make_tensor_value_info("x", tp.FLOAT, [2])],
                [h.make_tensor_value_info("y", tp.FLOAT, [2])],
            ),
            opset_imports=[h.make_opsetid("", 18)],
            ir_version=10,
        )
        (args.directory / "unknown.onnx").write_bytes(unknown.SerializeToString())
        (args.directory / "corrupt.onnx").write_bytes(b"\x0a\xff\xff")
        report["sha256"] = {
            str(p.relative_to(args.directory)): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in args.directory.rglob("*")
            if p.is_file()
        }
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
