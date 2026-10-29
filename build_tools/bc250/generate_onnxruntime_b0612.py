#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate modern static Torch exports and independent retained CPU/eager references."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import torch
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx.reference import ReferenceEvaluator
    from onnx_b0612_models import FAMILIES

    assert (
        ort.__version__ == "1.31.0"
        and "MIGraphXExecutionProvider" not in ort.get_available_providers()
    )
    assert (
        torch.__version__ == "2.13.0a0+gfx1013"
        and torch.cuda.get_device_properties(0).gcnArchName == "gfx1013:xnack-"
    )
    torch.set_num_threads(1)
    torch.manual_seed(6121)
    args.fixtures.mkdir()
    report = dict(
        status="running",
        checks=[],
        models={},
        tolerance=dict(rtol=3e-4, atol=2e-5),
        torch=torch.__version__,
        cpu_ort=ort.__version__,
        dynamo=True,
        static_batch=3,
    )

    def passed(name):
        report["checks"].append(name)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    try:
        for family, constructor, tail in FAMILIES:
            original = constructor().eval()
            reference = copy.deepcopy(original).double()
            for device in ("cpu", "cuda"):
                label = family + "-" + device
                folder = args.fixtures / label
                folder.mkdir()
                model = copy.deepcopy(original).to(device)
                before = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                sample = torch.zeros((3, *tail), device=device)
                program = torch.onnx.export(
                    model,
                    (sample,),
                    dynamo=True,
                    opset_version=18,
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
                assert all(n.domain in ("", "ai.onnx") for n in graph.graph.node)
                assert [
                    d.dim_value for d in graph.graph.input[0].type.tensor_type.shape.dim
                ] == [3, *tail]
                passed(label + "-dynamo-checked-static-export")
                so = ort.SessionOptions()
                so.intra_op_num_threads = so.inter_op_num_threads = 1
                so.enable_profiling = True
                so.profile_file_prefix = str(folder / "cpu-ort-profile")
                session = ort.InferenceSession(
                    str(file), sess_options=so, providers=["CPUExecutionProvider"]
                )
                session.disable_fallback()
                evaluator = ReferenceEvaluator(graph)
                outputs = []
                for replay in (0, 1):
                    data = (
                        np.random.default_rng(6121 + replay).normal(size=(3, *tail)) / 4
                    ).astype(np.float32)
                    with torch.no_grad():
                        gold = reference(torch.from_numpy(data).double()).numpy()
                        eager = model(torch.from_numpy(data).to(device)).cpu().numpy()
                        hip = (
                            eager
                            if device == "cuda"
                            else copy.deepcopy(original)
                            .cuda()(torch.from_numpy(data).cuda())
                            .cpu()
                            .numpy()
                        )
                    cpu = session.run(["y"], {"x": data})[0]
                    numpy = evaluator.run(["y"], {"x": data})[0]
                    for value in (eager, hip, cpu, numpy):
                        assert (
                            value.dtype == np.float32
                            and value.shape == gold.shape
                            and np.isfinite(value).all()
                        )
                        np.testing.assert_allclose(
                            value.astype(np.float64), gold, rtol=3e-4, atol=2e-5
                        )
                    data.tofile(folder / f"x-{replay}.f32")
                    gold.tofile(folder / f"cpu-{replay}.f64")
                    cpu.tofile(folder / f"ort-{replay}.f32")
                    hip.tofile(folder / f"hip-{replay}.f32")
                    numpy.astype(np.float32).tofile(folder / f"onnx-{replay}.f32")
                    outputs.append(gold)
                    passed(label + "-reference-set-" + str(replay))
                assert not np.array_equal(*outputs)
                profile = Path(session.end_profiling())
                events = json.loads(profile.read_text())
                nodes = [
                    e
                    for e in events
                    if e.get("cat") == "Node" and e.get("args", {}).get("provider")
                ]
                assert nodes and all(
                    e["args"]["provider"] == "CPUExecutionProvider" for e in nodes
                )
                passed(label + "-retained-cpu-provider-profile")
                assert all(
                    torch.equal(v.detach().cpu(), before[k])
                    for k, v in model.state_dict().items()
                )
                assert all(p.grad is None for p in model.parameters())
                np.savez(
                    folder / "weights.npz", **{k: v.numpy() for k, v in before.items()}
                )
                report["models"][label] = dict(
                    input_shape=[3, *tail],
                    output_shape=list(outputs[0].shape),
                    ir=graph.ir_version,
                    opsets={p.domain: p.version for p in graph.opset_import},
                    ops=[n.op_type for n in graph.graph.node],
                    export_device=device,
                    cpu_profile=str(profile),
                )
                passed(label + "-unchanged-weights-and-no-gradients")
        torch.cuda.synchronize()
        assert len(report["checks"]) == 30
        report["files_sha256"] = {
            str(p.relative_to(args.fixtures)): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in args.fixtures.rglob("*")
            if p.is_file()
        }
        report["status"] = "passed"
        (args.fixtures / "manifest.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
