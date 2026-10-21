#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Independent owned PNG samples, image codec interfaces and explicit HIP ops."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import struct
import zlib

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchcodec
from torchcodec.decoders import decode_image, decode_jpeg, decode_png, decode_webp
from torchcodec.encoders import JpegEncoder, PngEncoder


def chunk(kind, payload):
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload))
    )


def png_bytes(values, bits=8, exif=None):
    channels, height, width = values.shape
    color = {1: 0, 2: 4, 3: 2, 4: 6}[channels]
    rows = []
    for row in values.permute(1, 2, 0).tolist():
        flat = [int(v) for pixel in row for v in pixel]
        rows.append(
            b"\x00"
            + (bytes(flat) if bits == 8 else struct.pack(">" + "H" * len(flat), *flat))
        )
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, bits, color, 0, 0, 0))
        + (chunk(b"eXIf", exif) if exif else b"")
        + chunk(b"IDAT", zlib.compress(b"".join(rows)))
        + chunk(b"IEND", b"")
    )


def pil_tensor(source, mode=None):
    with Image.open(
        source if isinstance(source, Path) else io.BytesIO(source)
    ) as image:
        image.load()
        if mode:
            image = image.convert(mode)
        array = np.array(image, copy=True)
    if array.ndim == 2:
        array = array[:, :, None]
    return torch.from_numpy(array).permute(2, 0, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    assert str(torch.__version__) == "2.13.0a0+gfx1013"
    assert torchcodec.__version__ == "0.17.0+gfx1013"
    scratch = args.output.parent / "owned-images"
    scratch.mkdir()
    report = dict(
        status="running",
        checks=[],
        results=[],
        torch=str(torch.__version__),
        torchcodec=torchcodec.__version__,
        codec_device="cpu",
        gpu_codec_claim=False,
    )

    def persist():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def passed(name, **details):
        report["checks"].append(name)
        report["results"].append(dict(name=name, **details))
        persist()

    def exact(actual, expected):
        assert (
            actual.device.type == "cpu"
            and actual.shape == expected.shape
            and actual.dtype == expected.dtype
        )
        assert torch.equal(actual.long(), expected.long())

    def jpeg_close(actual, expected):
        assert (
            actual.device.type == "cpu"
            and actual.shape == expected.shape
            and actual.dtype == torch.uint8
        )
        assert int((actual.int() - expected.int()).abs().max()) <= 2

    try:
        h, w = torch.meshgrid(torch.arange(19), torch.arange(23), indexing="ij")
        gray = ((h * 5 + w * 3) % 256).to(torch.uint8).unsqueeze(0)
        rgb = torch.cat([((gray.int() + 37 * c) % 256).byte() for c in range(3)])
        alpha = ((h * 19 + w * 7) % 256).byte().unsqueeze(0)
        for image in (gray, torch.cat((gray, alpha)), rgb, torch.cat((rgb, alpha))):
            channels = image.shape[0]
            payload = png_bytes(image)
            path = scratch / f"independent-{channels}ch.png"
            path.write_bytes(payload)
            exact(decode_png(payload, mode="UNCHANGED"), image)
            passed(f"png{channels}-independent-bytes")
            exact(decode_image(path, mode="UNCHANGED"), image)
            passed(f"png{channels}-automatic-path")
            encoded = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
            exact(decode_png(encoded, mode="UNCHANGED"), image)
            passed(f"png{channels}-encoded-tensor")
            reference_rgb = image[:1].expand(3, -1, -1) if channels <= 2 else image[:3]
            exact(decode_png(path, mode="RGB"), reference_rgb)
            passed(f"png{channels}-rgb-mode")
            reference_alpha = (
                image[-1:] if channels in (2, 4) else torch.full_like(gray, 255)
            )
            exact(
                decode_png(path, mode="RGB_ALPHA"),
                torch.cat((reference_rgb, reference_alpha)),
            )
            passed(f"png{channels}-rgba-mode")
            exact(
                decode_png(path, mode="UNCHANGED", output_dtype=torch.uint16),
                (image.int() * 257).to(torch.uint16),
            )
            passed(f"png{channels}-uint16-promotion")
        for channels in (1, 3):
            values = ((rgb[:channels].long() * 251 + h * 17 + w * 31) % 65536).to(
                torch.uint16
            )
            path = scratch / f"independent16-{channels}ch.png"
            path.write_bytes(png_bytes(values, bits=16))
            exact(decode_png(path, mode="UNCHANGED", output_dtype="auto"), values)
            passed(f"png16-{channels}-native-auto")
            exact(
                decode_image(path, mode="UNCHANGED", output_dtype=torch.uint16), values
            )
            passed(f"png16-{channels}-automatic-preserved")
            expected = (values.float() / 257).round().clamp(0, 255).byte()
            exact(decode_png(path, mode="UNCHANGED"), expected)
            passed(f"png16-{channels}-uint8-scaling")
        exif = b"II\x2a\x00" + struct.pack("<IHHHIHHI", 8, 1, 0x112, 3, 1, 6, 0, 0)
        oriented = png_bytes(rgb, exif=exif)
        exact(decode_png(oriented), torch.rot90(rgb, -1, (1, 2)))
        passed("png-independent-exif-orientation6")
        for image in (gray, rgb):
            mode = "L" if image.shape[0] == 1 else "RGB"
            array = image.permute(1, 2, 0).numpy()
            if mode == "L":
                array = array[:, :, 0]
            path = scratch / (mode + ".jpg")
            Image.fromarray(array).save(path, quality=95, subsampling=0)
            payload = path.read_bytes()
            reference = pil_tensor(path, mode)
            for kind, source in (
                ("path", path),
                ("bytes", payload),
                ("tensor", torch.frombuffer(bytearray(payload), dtype=torch.uint8)),
            ):
                jpeg_close(decode_jpeg(source, mode="UNCHANGED"), reference)
                passed(f"jpeg-{mode}-{kind}-versus-pillow")
            jpeg_close(decode_image(path, mode="UNCHANGED"), reference)
            passed(f"jpeg-{mode}-automatic-versus-pillow")
            actual = decode_jpeg(path, mode="UNCHANGED")
            exact(
                decode_jpeg(path, mode="UNCHANGED", output_dtype=torch.uint16),
                (actual.int() * 257).to(torch.uint16),
            )
            passed(f"jpeg-{mode}-uint16-promotion")
        for channels in (3, 4):
            # Preserve meaningful RGB at every alpha; no invisible-color claim.
            image = rgb if channels == 3 else torch.cat((rgb, alpha.clamp_min(1)))
            path = scratch / f"lossless-{channels}.webp"
            Image.fromarray(image.permute(1, 2, 0).numpy()).save(
                path, lossless=True, exact=True
            )
            payload = path.read_bytes()
            for kind, source in (
                ("path", path),
                ("bytes", payload),
                ("tensor", torch.frombuffer(bytearray(payload), dtype=torch.uint8)),
            ):
                exact(decode_webp(source, mode="UNCHANGED"), image)
                passed(f"webp-{channels}-{kind}-independent-pixels")
            exact(decode_image(path, mode="UNCHANGED"), image)
            passed(f"webp-{channels}-automatic")
        path = scratch / "animated.webp"
        images = [
            Image.fromarray(v.permute(1, 2, 0).numpy()) for v in (rgb, rgb.flip(-1))
        ]
        images[0].save(
            path,
            save_all=True,
            append_images=images[1:],
            lossless=True,
            duration=[40, 70],
            loop=0,
        )
        reference = torch.stack((rgb, rgb.flip(-1)))
        exact(decode_webp(path), reference)
        passed("webp-two-frame-animation-pixels")
        exact(decode_image(path), reference)
        passed("webp-animation-automatic")
        for extension, encoder_cls in (("png", PngEncoder), ("jpg", JpegEncoder)):
            for image in (gray, rgb):
                for view in ("contiguous", "strided"):
                    original = image if view == "contiguous" else image[..., ::2]
                    for target in ("path", "filelike", "tensor"):
                        encoder = encoder_cls(original)
                        kwargs = (
                            {"quality": 95}
                            if extension == "jpg"
                            else {"compression_level": 6}
                        )
                        path = (
                            scratch
                            / f"encoded-{extension}-{original.shape[0]}-{view}-{target}.{extension}"
                        )
                        if target == "path":
                            encoder.to_file(path, **kwargs)
                            payload = path.read_bytes()
                        elif target == "filelike":
                            stream = io.BytesIO()
                            encoder.to_file_like(stream, **kwargs)
                            payload = stream.getvalue()
                            path.write_bytes(payload)
                        else:
                            data = encoder.to_tensor(**kwargs)
                            assert (
                                data.device.type == "cpu"
                                and data.dtype == torch.uint8
                                and data.ndim == 1
                            )
                            payload = bytes(data.tolist())
                            path.write_bytes(payload)
                        reference = pil_tensor(
                            payload, "L" if image.shape[0] == 1 else "RGB"
                        )
                        if extension == "png":
                            exact(reference, original)
                            exact(decode_image(payload, mode="UNCHANGED"), original)
                        else:
                            jpeg_close(
                                decode_jpeg(payload, mode="UNCHANGED"), reference
                            )
                            assert (
                                float(
                                    (reference.float() - original.float()).abs().mean()
                                )
                                < 20
                            )
                        passed(f"encode-{extension}-{image.shape[0]}-{view}-{target}")
        for name, fn in (
            ("corrupt-png", lambda: decode_png(b"\x89PNG\r\n\x1a\n" + b"broken" * 5)),
            ("corrupt-jpeg", lambda: decode_jpeg(b"\xff\xd8broken")),
            ("corrupt-webp", lambda: decode_webp(b"RIFF\x10\x00\x00\x00WEBPbroken")),
            ("invalid-mode", lambda: decode_png(oriented, mode="invalid")),
            ("encoded-dtype", lambda: decode_png(torch.zeros(10, dtype=torch.float32))),
            ("jpeg-quality", lambda: JpegEncoder(rgb).to_tensor(quality=0)),
            ("png-rank", lambda: PngEncoder(rgb.unsqueeze(0)).to_tensor()),
            ("png-channels", lambda: PngEncoder(torch.cat((rgb, alpha))).to_tensor()),
        ):
            try:
                fn()
            except (RuntimeError, ValueError, TypeError):
                passed("refuse-" + name)
            else:
                raise AssertionError("Expected rejection: " + name)
        try:
            decode_jpeg(scratch / "RGB.jpg", device="cuda")
        except (RuntimeError, ValueError) as error:
            assert any(
                v in str(error).lower() for v in ("cuda", "nvjpeg", "not compiled")
            )
            passed("explicit-unavailable-gpu-jpeg")
        else:
            raise AssertionError("Unexpected GPU JPEG backend")
        decoded = decode_image(scratch / "independent-3ch.png")
        cpu = (decoded.double() / 255).unsqueeze(0).requires_grad_()
        gpu = (decoded.cuda().float() / 255).unsqueeze(0).requires_grad_()
        reference = F.avg_pool2d(cpu, 2)
        actual = F.avg_pool2d(gpu, 2)
        weights = torch.linspace(
            -0.6, 0.9, reference.numel(), dtype=torch.float64
        ).reshape(reference.shape)
        gc = torch.autograd.grad((reference * weights).sum(), cpu)[0]
        gg = torch.autograd.grad((actual * weights.cuda().float()).sum(), gpu)[0]
        torch.cuda.synchronize()
        assert actual.is_cuda and gg.is_cuda
        torch.testing.assert_close(
            actual.cpu().double(), reference, rtol=2e-6, atol=2e-7
        )
        torch.testing.assert_close(gg.cpu().double(), gc, rtol=2e-6, atol=2e-7)
        passed("cpu-image-to-hip-spatial-forward-backward")
        output = (actual.detach().cpu()[0] * 255).round().clamp(0, 255).byte()
        path = scratch / "processed.png"
        PngEncoder(output).to_file(path)
        exact(pil_tensor(path), output)
        passed("hip-result-explicit-cpu-png-independent-pixels")
        mapped = set()
        for line in Path("/proc/self/maps").read_text().splitlines():
            name = line.split()[-1]
            if name.startswith("/") and any(
                v in name
                for v in (
                    "libtorchcodec",
                    "libjpeg",
                    "libpng",
                    "libwebp",
                    "libsharpyuv",
                )
            ):
                mapped.add(name)
        report["mapped_codec_sha256"] = {
            n: hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in sorted(mapped)
        }
        assert any("/opt/bc250-codec-sysdeps/lib/libwebpdemux" in n for n in mapped)
        report["fixture_sha256"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in scratch.iterdir()
            if p.is_file()
        }
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        persist()


if __name__ == "__main__":
    main()
