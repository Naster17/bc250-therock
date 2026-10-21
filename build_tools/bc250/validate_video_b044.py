#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Owned lossless CPU video references, seeking, encoding and HIP interop."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess

import torch
import torch.nn.functional as F
import torchcodec
from torchcodec.decoders import VideoDecoder
from torchcodec.encoders import Encoder, VideoEncoder


def ffmpeg(*args):
    return subprocess.check_output(
        ["ffmpeg", "-nostdin", "-v", "error", "-threads", "2", *map(str, args)],
        stderr=subprocess.PIPE,
        timeout=30,
    )


def raw_decode(path):
    return ffmpeg(
        "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-threads", "2", "pipe:1"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    assert str(torch.__version__) == "2.13.0a0+gfx1013"
    assert torchcodec.__version__ == "0.17.0+gfx1013"
    scratch = args.output.parent / "owned-video"
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
        assert actual.device.type == "cpu" and actual.dtype == torch.uint8
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def timestamps(actual, expected):
        torch.testing.assert_close(
            torch.as_tensor(actual, dtype=torch.float64),
            torch.as_tensor(expected, dtype=torch.float64),
            rtol=0,
            atol=1e-9,
        )

    try:
        n, c, h, w = torch.meshgrid(
            torch.arange(9),
            torch.arange(3),
            torch.arange(18),
            torch.arange(26),
            indexing="ij",
        )
        frames = ((n * 37 + c * 61 + h * 13 + w * 7 + n * w * 3) % 256).to(torch.uint8)
        packed = frames.permute(0, 2, 3, 1).contiguous().numpy().tobytes()
        raw = scratch / "reference.rgb24"
        raw.write_bytes(packed)
        input_args = [
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            "26x18",
            "-framerate",
            "5",
            "-i",
            raw,
        ]
        paths = []
        for codec, pixel in (("ffv1", "bgr0"), ("libx264rgb", "rgb24")):
            path = scratch / (codec + ".mkv")
            options = ["-c:v", codec, "-pix_fmt", pixel, "-threads", "2"]
            if codec == "libx264rgb":
                options += ["-crf", "0", "-preset", "medium", "-g", "6", "-bf", "2"]
            ffmpeg(*input_args, *options, path)
            assert raw_decode(path) == packed
            paths.append(path)
            probe = json.loads(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_frames",
                        "-show_entries",
                        "frame=pict_type",
                        "-of",
                        "json",
                        str(path),
                    ],
                    timeout=30,
                )
            )
            report.setdefault("fixture_picture_types", {})[codec] = [
                f["pict_type"] for f in probe["frames"]
            ]
            for order in ("NCHW", "NHWC"):
                expected = frames if order == "NCHW" else frames.permute(0, 2, 3, 1)
                decoder = VideoDecoder(
                    path, device="cpu", dimension_order=order, num_ffmpeg_threads=2
                )
                label = codec + "-" + order
                assert (
                    len(decoder) == 9
                    and decoder.metadata.width == 26
                    and decoder.metadata.height == 18
                )
                passed(label + "-metadata")
                batch = decoder.get_all_frames()
                exact(batch.data, expected)
                timestamps(batch.pts_seconds, torch.arange(9, dtype=torch.float64) / 5)
                timestamps(batch.duration_seconds, [0.2] * 9)
                passed(label + "-all-pixels-pts-duration")
                for i in (8, 1, 7, 0):
                    frame = decoder.get_frame_at(i)
                    exact(frame.data, expected[i])
                    timestamps(frame.pts_seconds, i / 5)
                exact(decoder[-1], expected[-1])
                passed(label + "-backward-forward-seeks-and-negative-index")
                indices = [8, 1, 4, 4, 0]
                batch = decoder.get_frames_at(indices)
                exact(batch.data, expected[indices])
                timestamps(batch.pts_seconds, [i / 5 for i in indices])
                passed(label + "-unsorted-duplicate-batch")
                exact(decoder.get_frames_in_range(1, 9, 2).data, expected[1:9:2])
                passed(label + "-index-range")
                exact(decoder[2:8:2], expected[2:8:2])
                passed(label + "-slice")
                for i in (7, 0, 3):
                    exact(
                        decoder.get_frame_played_at((i + 0.375) / 5).data, expected[i]
                    )
                passed(label + "-timestamp-seek")
                exact(
                    decoder.get_frames_played_at(
                        [(i + 0.375) / 5 for i in indices]
                    ).data,
                    expected[indices],
                )
                passed(label + "-timestamp-batch")
            for kind, source in (
                ("bytes", path.read_bytes()),
                ("filelike", io.BytesIO(path.read_bytes())),
                (
                    "tensor",
                    torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.uint8),
                ),
            ):
                decoder = VideoDecoder(source, device="cpu", num_ffmpeg_threads=2)
                exact(decoder.get_frames_at([8, 0, 5]).data, frames[[8, 0, 5]])
                passed(codec + "-source-" + kind)
        # Variable presentation timestamps are prescribed from frame index,
        # independently of the decoder: t_i=i*i/5, with nine preserved frames.
        vfr = scratch / "variable-pts.mkv"
        ffmpeg(
            *input_args,
            "-vf",
            "setpts=N*N/(5*TB)",
            "-fps_mode",
            "vfr",
            "-c:v",
            "ffv1",
            "-pix_fmt",
            "bgr0",
            vfr,
        )
        decoder = VideoDecoder(vfr, device="cpu")
        batch = decoder.get_all_frames()
        exact(batch.data, frames)
        times = [i * i / 5 for i in range(9)]
        timestamps(batch.pts_seconds, times)
        passed("variable-pts-all-pixels-and-independent-pts")
        exact(decoder.get_frames_at([8, 1, 6, 0]).data, frames[[8, 1, 6, 0]])
        passed("variable-pts-index-seeks")
        for i in (6, 0, 3, 1):
            exact(
                decoder.get_frame_played_at((times[i] + times[i + 1]) / 2).data,
                frames[i],
            )
        passed("variable-pts-time-seeks")
        # Independent FFmpeg CLI decoding verifies encoder pixels and frame count.
        for codec, pixel in (("ffv1", "bgr0"), ("libx264rgb", "rgb24")):
            for view in ("contiguous", "strided"):
                source = frames if view == "contiguous" else frames[..., ::2]
                path = scratch / f"encoded-{codec}-{view}.mkv"
                options = dict(codec=codec, pixel_format=pixel)
                if codec == "libx264rgb":
                    options.update(crf=0, preset="medium")
                VideoEncoder(source, frame_rate=5).to_file(path, **options)
                assert (
                    raw_decode(path)
                    == source.permute(0, 2, 3, 1).contiguous().numpy().tobytes()
                )
                exact(VideoDecoder(path, device="cpu").get_all_frames().data, source)
                passed(f"encode-{codec}-{view}-independent-pixels")
        for kind in ("tensor", "filelike"):
            encoder = VideoEncoder(frames, frame_rate=5)
            if kind == "tensor":
                source = encoder.to_tensor(
                    "matroska", codec="ffv1", pixel_format="bgr0"
                )
            else:
                source = io.BytesIO()
                encoder.to_file_like(
                    source, "matroska", codec="ffv1", pixel_format="bgr0"
                )
                source.seek(0)
            exact(VideoDecoder(source, device="cpu").get_all_frames().data, frames)
            passed("encode-" + kind + "-decode-exact")
        path = scratch / "incremental.mkv"
        encoder = Encoder()
        stream = encoder.add_video(
            height=18, width=26, frame_rate=5, codec="ffv1", pixel_format="bgr0"
        )
        with encoder.open_file(path):
            stream.add_frames(frames[:4])
            stream.add_frames(frames[4:])
        assert raw_decode(path) == packed
        passed("incremental-two-chunk-encode-independent-pixels")
        for name, fn in (
            (
                "missing-file",
                lambda: VideoDecoder(scratch / "missing.mkv", device="cpu"),
            ),
            (
                "invalid-order",
                lambda: VideoDecoder(paths[0], dimension_order="invalid"),
            ),
            ("invalid-index", lambda: decoder.get_frame_at(9)),
            ("invalid-dtype", lambda: VideoEncoder(frames.float(), frame_rate=5)),
            ("zero-frame-rate", lambda: VideoEncoder(frames, frame_rate=0)),
        ):
            try:
                fn()
            except (ValueError, RuntimeError, IndexError):
                passed("refuse-" + name)
            else:
                raise AssertionError("Expected rejection: " + name)
        # CPU-decoded frames -> GPU differentiable spatial operation -> CPU
        # encoding. The file boundary itself is not differentiable.
        decoded = VideoDecoder(paths[0], device="cpu").get_all_frames().data
        cpu = (decoded.double() / 255).requires_grad_()
        gpu = (decoded.cuda().float() / 255).requires_grad_()
        reference = F.avg_pool2d(cpu, 2)
        result = F.avg_pool2d(gpu, 2)
        weights = torch.linspace(
            -0.7, 0.8, reference.numel(), dtype=torch.float64
        ).reshape(reference.shape)
        grad_cpu = torch.autograd.grad((reference * weights).sum(), cpu)[0]
        grad_gpu = torch.autograd.grad((result * weights.cuda().float()).sum(), gpu)[0]
        torch.cuda.synchronize()
        assert result.is_cuda and grad_gpu.is_cuda
        torch.testing.assert_close(
            result.cpu().double(), reference, rtol=2e-6, atol=2e-7
        )
        torch.testing.assert_close(
            grad_gpu.cpu().double(), grad_cpu, rtol=2e-6, atol=2e-7
        )
        passed("cpu-video-to-hip-spatial-forward-backward")
        output = (result.detach().cpu() * 255).round().clamp(0, 255).to(torch.uint8)
        path = scratch / "processed.mkv"
        VideoEncoder(output, frame_rate=5).to_file(
            path, codec="ffv1", pixel_format="bgr0"
        )
        assert (
            raw_decode(path)
            == output.permute(0, 2, 3, 1).contiguous().numpy().tobytes()
        )
        passed("hip-result-explicit-cpu-encode-independent-pixels")
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
