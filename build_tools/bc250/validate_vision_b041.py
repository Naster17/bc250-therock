#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Native torchvision operators versus CPU FP64 and independent small oracles."""

import argparse
import io
import json
import math
from pathlib import Path

import torch
import torchvision as tv
from torch.nn import functional as F


def nms_reference(boxes, scores, threshold):
    order = scores.argsort(descending=True).tolist()
    keep = []
    while order:
        first = order.pop(0)
        keep.append(first)
        remaining = []
        a = boxes[first]
        for index in order:
            b = boxes[index]
            intersection = (
                (torch.minimum(a[2:], b[2:]) - torch.maximum(a[:2], b[:2]))
                .clamp_min(0)
                .prod()
            )
            union = (a[2:] - a[:2]).prod() + (b[2:] - b[:2]).prod() - intersection
            if float(intersection / union) <= threshold:
                remaining.append(index)
        order = remaining
    return torch.tensor(keep, dtype=torch.long)


def roi_reference(x, rois, pooled, aligned, ratio=2):
    outputs = []
    _, channels, height, width = x.shape
    for roi in rois:
        batch = int(roi[0])
        start_x, start_y, end_x, end_y = [
            float(v) - (0.5 if aligned else 0) for v in roi[1:]
        ]
        rw = end_x - start_x
        rh = end_y - start_y
        if not aligned:
            rw = max(rw, 1.0)
            rh = max(rh, 1.0)
        gw = ratio if ratio > 0 else math.ceil(rw / pooled[1])
        gh = ratio if ratio > 0 else math.ceil(rh / pooled[0])
        rows = []
        for py in range(pooled[0]):
            cols = []
            for px in range(pooled[1]):
                value = x[batch, :, 0, 0] * 0
                for iy in range(gh):
                    for ix in range(gw):
                        y = start_y + (py + (iy + 0.5) / gh) * rh / pooled[0]
                        z = start_x + (px + (ix + 0.5) / gw) * rw / pooled[1]
                        if y < -1 or y > height or z < -1 or z > width:
                            continue
                        y = max(y, 0.0)
                        z = max(z, 0.0)
                        yl = min(int(y), height - 1)
                        xl = min(int(z), width - 1)
                        yh = min(yl + 1, height - 1)
                        xh = min(xl + 1, width - 1)
                        if yl == height - 1:
                            y = float(yl)
                        if xl == width - 1:
                            z = float(xl)
                        dy = y - yl
                        dx = z - xl
                        value = (
                            value
                            + x[batch, :, yl, xl] * (1 - dy) * (1 - dx)
                            + x[batch, :, yl, xh] * (1 - dy) * dx
                            + x[batch, :, yh, xl] * dy * (1 - dx)
                            + x[batch, :, yh, xh] * dy * dx
                        )
                cols.append(value / (gh * gw))
            rows.append(torch.stack(cols, -1))
        outputs.append(torch.stack(rows, -2))
    return torch.stack(outputs) if outputs else x.new_empty((0, channels, *pooled))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group", choices=("ops", "integration"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(4041)
    assert str(tv.__version__) == "0.28.0+gfx1013" and tv.extension._has_ops()
    assert (
        torch.__version__ == "2.13.0a0+gfx1013"
        and torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1013"
    )
    report = dict(
        status="running",
        checks=[],
        results=[],
        torch=str(torch.__version__),
        torchvision=str(tv.__version__),
        torchvision_git=tv.version.git_version,
        torchvision_file=tv.__file__,
        group=args.group,
    )

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def passed(name, **details):
        torch.cuda.synchronize()
        report["checks"].append(name)
        report["results"].append(dict(name=name, **details))
        save()

    def close(a, b, dtype):
        a = a.detach().cpu().double()
        b = b.detach().double()
        assert a.isfinite().all() and b.isfinite().all()
        tol = {
            torch.float64: (1e-9, 1e-10, 1e-8),
            torch.float32: (0.002, 2e-5, 0.003),
            torch.float16: (0.05, 0.008, 0.04),
        }[dtype]
        torch.testing.assert_close(a, b, rtol=tol[0], atol=tol[1])
        relative = float((a - b).norm() / b.norm().clamp_min(1e-12))
        assert relative < tol[2], relative
        return relative

    def run(name, fn, inputs, flags, dtype, strided=False, trace=False):
        cpu = [
            x.detach().double().clone().requires_grad_(flag)
            for x, flag in zip(inputs, flags)
        ]
        expected = fn(*cpu)
        up = (
            torch.linspace(-0.8, 0.9, expected.numel())
            .reshape(expected.shape)
            .to(dtype)
        )
        expected.backward(up.double())
        gpu = []
        for x, flag in zip(inputs, flags):
            if strided and x.ndim == 4:
                storage = torch.empty(
                    (*x.shape[:-1], x.shape[-1] * 2), dtype=dtype, device="cuda"
                )
                y = storage[..., ::2]
                y.copy_(x)
            else:
                y = x.cuda()
            gpu.append(y.detach().requires_grad_(flag))
        if trace:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                actual = fn(*gpu)
                actual.backward(up.cuda())
                torch.cuda.synchronize()
            path = args.output.parent / "vision-native-trace.json"
            prof.export_chrome_trace(str(path))
            events = json.loads(path.read_text())["traceEvents"]
            kernels = [e for e in events if e.get("cat") == "kernel"]
            assert kernels and any("roi" in e["name"].lower() for e in kernels)
            report["gpu_kernel_names"] = sorted({e["name"] for e in kernels})
        else:
            actual = fn(*gpu)
            actual.backward(up.cuda())
        errors = [close(actual, expected, dtype)]
        for a, b, flag in zip(gpu, cpu, flags):
            if flag:
                assert a.grad is not None and b.grad is not None
                errors.append(close(a.grad, b.grad, dtype))
        passed(
            name,
            max_relative_l2=max(errors),
            gradient_tensors=sum(flags),
            noncontiguous=strided,
            trace=trace,
        )

    try:
        if args.group == "ops":
            for dtype in (torch.float32, torch.float64, torch.float16):
                for n in (0, 5, 73):
                    p = torch.rand(n, 2) * 5
                    boxes = torch.cat((p, p + torch.rand(n, 2) * 3 + 0.2), 1).to(dtype)
                    scores = torch.linspace(0.99, 0.01, n).to(dtype)
                    expected = nms_reference(boxes.double(), scores.double(), 0.35)
                    actual = tv.ops.nms(boxes.cuda(), scores.cuda(), 0.35).cpu()
                    assert torch.equal(actual, expected)
                    passed(f"nms-{dtype}-{n}")
                categories = torch.arange(n) % 3
                expected = []
                for category in range(3):
                    ids = (categories == category).nonzero().flatten()
                    expected.extend(
                        ids[
                            nms_reference(
                                boxes[ids].double(), scores[ids].double(), 0.35
                            )
                        ].tolist()
                    )
                expected = sorted(
                    expected, key=lambda k: float(scores[k]), reverse=True
                )
                assert (
                    tv.ops.batched_nms(
                        boxes.cuda(), scores.cuda(), categories.cuda(), 0.35
                    )
                    .cpu()
                    .tolist()
                    == expected
                )
                passed(f"batched-nms-{dtype}")
                for kind in ("roi_align", "roi_pool", "ps_roi_align", "ps_roi_pool"):
                    for layout in (0, 1):
                        x = (torch.randn(2, 8, 7, 9) * 0.2).to(dtype)
                        rois = torch.tensor(
                            [[0, 0.25, 0.5, 6.75, 5.5], [1, 1.25, 0.25, 7.5, 6.0]],
                            dtype=dtype,
                        )
                        if kind == "roi_align":
                            fn = lambda a, r: tv.ops.roi_align(
                                a, r, (2, 2), sampling_ratio=2, aligned=bool(layout)
                            )
                        elif kind == "roi_pool":
                            fn = lambda a, r: tv.ops.roi_pool(a, r, (2, 2))
                        elif kind == "ps_roi_align":
                            fn = lambda a, r: tv.ops.ps_roi_align(
                                a, r, (2, 2), sampling_ratio=2
                            )
                        else:
                            fn = lambda a, r: tv.ops.ps_roi_pool(a, r, (2, 2))
                        run(
                            f"{kind}-{dtype}-layout{layout}",
                            fn,
                            [x, rois],
                            [True, False],
                            dtype,
                            strided=bool(layout),
                        )
            for kind in ("roi_align", "roi_pool", "ps_roi_align", "ps_roi_pool"):
                x = torch.randn(2, 8, 7, 9)
                rois = torch.empty(0, 5)
                fn = getattr(tv.ops, kind)
                gx = x.cuda().requires_grad_()
                y = fn(gx, rois.cuda(), (2, 2))
                assert y.shape[0] == 0
                y.sum().backward()
                assert gx.grad is not None and torch.count_nonzero(gx.grad) == 0
                passed(kind + "-empty-forward-backward")
            for aligned, ratio in ((False, 2), (True, 2), (True, 0)):
                x = torch.randn(2, 3, 5, 7, dtype=torch.float64, requires_grad=True)
                rois = torch.tensor(
                    [[0, 0.25, 0.5, 5.75, 4.5], [1, -0.3, -0.2, 6.0, 4.0]],
                    dtype=torch.float64,
                )
                a = tv.ops.roi_align(
                    x, rois, (2, 3), sampling_ratio=ratio, aligned=aligned
                )
                b = roi_reference(x, rois, (2, 3), aligned, ratio)
                up = torch.linspace(-0.8, 0.9, a.numel()).reshape(a.shape).double()
                ga = torch.autograd.grad(a, x, up, retain_graph=True)[0]
                gb = torch.autograd.grad(b, x, up)[0]
                torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
                torch.testing.assert_close(ga, gb, atol=1e-12, rtol=1e-12)
                passed(f"independent-bilinear-oracle-aligned{aligned}-ratio{ratio}")
            for dtype in (torch.float32, torch.float64, torch.float16):
                for masked in (False, True):
                    x = (torch.randn(1, 4, 5, 6) * 0.2).to(dtype)
                    weight = (torch.randn(6, 2, 3, 3) * 0.1).to(dtype)
                    offset = (torch.rand(1, 36, 5, 6) * 0.2 + 0.05).to(dtype)
                    bias = (torch.randn(6) * 0.1).to(dtype)
                    if masked:
                        mask = (torch.rand(1, 18, 5, 6) * 0.6 + 0.2).to(dtype)
                        fn = lambda a, o, w, b, m: tv.ops.deform_conv2d(
                            a, o, w, b, padding=(1, 1), mask=m
                        )
                        inputs = [x, offset, weight, bias, mask]
                    else:
                        fn = lambda a, o, w, b: tv.ops.deform_conv2d(
                            a, o, w, b, padding=(1, 1)
                        )
                        inputs = [x, offset, weight, bias]
                    run(
                        f"deform-{dtype}-mask{masked}",
                        fn,
                        inputs,
                        [True] * len(inputs),
                        dtype,
                    )
            x = torch.randn(1, 4, 5, 6, dtype=torch.float64, requires_grad=True)
            w = torch.randn(6, 2, 3, 3, dtype=torch.float64, requires_grad=True)
            o = torch.zeros(1, 36, 5, 6, dtype=torch.float64)
            a = tv.ops.deform_conv2d(x, o, w, padding=(1, 1))
            b = F.conv2d(x, w, padding=1, groups=2)
            torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
            passed("zero-offset-deform-equals-convolution")
            run(
                "profiled-native-roi-output-backward",
                lambda a, r: tv.ops.roi_align(a, r, (2, 3), sampling_ratio=2),
                [torch.randn(2, 4, 5, 7), torch.tensor([[0, 0.2, 0.3, 5.5, 4.0]])],
                [True, False],
                torch.float32,
                trace=True,
            )
        else:
            from PIL import Image
            import numpy as np

            image = (
                torch.arange(3 * 11 * 13, dtype=torch.int64)
                .remainder(251)
                .byte()
                .reshape(3, 11, 13)
            )
            for codec in ("png", "jpeg", "webp"):
                if codec == "png":
                    encoded = tv.io.encode_png(image)
                    decoded = tv.io.decode_png(encoded)
                    torch.testing.assert_close(decoded, image, rtol=0, atol=0)
                elif codec == "jpeg":
                    encoded = tv.io.encode_jpeg(image, quality=90)
                    decoded = tv.io.decode_jpeg(encoded)
                else:
                    data = io.BytesIO()
                    Image.fromarray(image.permute(1, 2, 0).numpy()).save(
                        data, format="WEBP", lossless=True
                    )
                    encoded = torch.tensor(list(data.getvalue()), dtype=torch.uint8)
                    decoded = tv.io.decode_webp(encoded)
                reference = torch.from_numpy(
                    np.array(Image.open(io.BytesIO(bytes(encoded.tolist()))))
                ).permute(2, 0, 1)
                torch.testing.assert_close(
                    decoded.float(),
                    reference.float(),
                    atol=3 if codec == "jpeg" else 0,
                    rtol=0,
                )
                assert torch.equal(decoded.cuda().cpu(), decoded)
                passed("cpu-codec-and-hip-transfer-" + codec)
            x = torch.randn(2, 3, 17, 19, dtype=torch.float64, requires_grad=True)
            for name, fn in (
                (
                    "resize",
                    lambda a: tv.transforms.functional.resize(
                        a, [11, 13], antialias=True
                    ),
                ),
                (
                    "normalize",
                    lambda a: tv.transforms.functional.normalize(
                        a, [0.1, 0.2, 0.3], [0.7, 0.8, 0.9]
                    ),
                ),
                (
                    "affine",
                    lambda a: tv.transforms.functional.affine(
                        a, 12, [1, -1], 0.9, [0.0, 0.0]
                    ),
                ),
                ("crop", lambda a: tv.transforms.functional.center_crop(a, [11, 13])),
            ):
                run(
                    "transform-" + name,
                    lambda a: fn(a),
                    [x.float()],
                    [True],
                    torch.float32,
                )
            torch.manual_seed(4042)
            cpu = tv.models.resnet18(weights=None, num_classes=7).eval()
            gpu = tv.models.resnet18(weights=None, num_classes=7).eval().cuda()
            gpu.load_state_dict(cpu.state_dict())
            x = torch.randn(2, 3, 32, 32) * 0.1
            cx = x.detach().clone().requires_grad_()
            gx = x.cuda().requires_grad_()
            a = cpu(cx)
            b = gpu(gx)
            up = torch.linspace(-0.8, 0.9, a.numel()).reshape(a.shape)
            a.backward(up)
            b.backward(up.cuda())
            close(b, a, torch.float32)
            close(gx.grad, cx.grad, torch.float32)
            for (_, p), (_, q) in zip(gpu.named_parameters(), cpu.named_parameters()):
                close(p.grad, q.grad, torch.float32)
            passed(
                "synthetic-resnet18-output-input-all-parameter-gradients",
                parameter_tensors=len(list(cpu.parameters())),
                weights_downloaded=False,
            )
            import importlib.metadata as metadata

            assert (
                torch.__version__ == "2.13.0a0+gfx1013"
                and torch.version.hip == "7.17.0"
            )
            passed(
                "qualified-torch-version-native-ops-import",
                pillow=metadata.version("pillow"),
            )
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
