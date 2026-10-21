#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Owned CPU audio codec fixtures plus explicit HIP signal interoperability."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import warnings
import wave

import torch
import torchaudio as ta
import torchcodec
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder


def cli(*arguments):
    return subprocess.check_output(
        ["ffmpeg", "-nostdin", "-v", "error", *map(str, arguments)],
        stderr=subprocess.PIPE,
        timeout=30,
    )


def write_pcm(path, values, rate, width):
    integers = values.T.reshape(-1).tolist()
    payload = b"".join(int(v).to_bytes(width, "little", signed=True) for v in integers)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(values.shape[0])
        stream.setsampwidth(width)
        stream.setframerate(rate)
        stream.writeframes(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    assert str(torch.__version__) == "2.13.0a0+gfx1013"
    assert ta.__version__ == "2.11.0+gfx1013"
    assert torchcodec.__version__ == "0.17.0+gfx1013"
    scratch = args.output.parent / "owned-fixtures"
    scratch.mkdir()
    report = dict(
        status="running",
        checks=[],
        results=[],
        torch=str(torch.__version__),
        torchaudio=ta.__version__,
        torchcodec=torchcodec.__version__,
        ffmpeg=subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[
            0
        ],
        decoding_device="cpu",
        gpu_codec_claim=False,
    )

    def persist():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def passed(name, **details):
        report["checks"].append(name)
        report["results"].append(dict(name=name, **details))
        persist()

    def exact(actual, expected):
        assert actual.device.type == "cpu" and actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected.float(), rtol=0, atol=0)

    try:
        # Independent integer PCM waveform, channel order and RIFF serialization.
        n = torch.arange(4003, dtype=torch.int64)
        stereo = torch.stack(
            ((n * 193 + 97) % 50001 - 25000, (n * 367 + 701) % 48001 - 24000)
        )
        fixtures = []
        for channels, rate, width in ((1, 8000, 2), (2, 16000, 2), (2, 16000, 3)):
            scale = 1 << (8 * width - 1)
            integers = stereo[:channels] * (251 if width == 3 else 1)
            path = scratch / f"pcm{width * 8}-{channels}ch-{rate}.wav"
            write_pcm(path, integers, rate, width)
            fixtures.append((path, integers.float() / scale, rate))
        pcm_path, pcm, rate = fixtures[1]
        flac = scratch / "independent-reference.flac"
        cli("-y", "-i", pcm_path, "-c:a", "flac", flac)
        fixtures.append((flac, pcm, rate))
        for path, expected, rate in fixtures:
            label = path.stem
            data, sr = ta.load(path)
            exact(data, expected)
            assert sr == rate
            passed(label + "-load-path")
            data, sr = ta.load(io.BytesIO(path.read_bytes()))
            exact(data, expected)
            assert sr == rate
            passed(label + "-load-filelike")
            data, sr = ta.load(path, channels_first=False)
            exact(data, expected.T)
            assert sr == rate
            passed(label + "-channels-last")
            for layout in (True, False):
                data, sr = ta.load(
                    path, frame_offset=37, num_frames=113, channels_first=layout
                )
                exact(data, expected[:, 37:150] if layout else expected[:, 37:150].T)
                assert sr == rate
                passed(label + f"-crop-channels-first-{layout}")
            for offset, count in ((0, 0), (5000, -1)):
                data, sr = ta.load(path, frame_offset=offset, num_frames=count)
                exact(data, expected[:, :0])
                assert sr == rate
                passed(label + f"-empty-offset{offset}-count{count}")
            decoder = AudioDecoder(path)
            samples = decoder.get_all_samples()
            exact(samples.data, expected)
            assert samples.sample_rate == rate
            assert decoder.metadata.num_channels == expected.shape[0]
            passed(label + "-direct-decoder-metadata")
        # A lossy decoder is compared with the independent FFmpeg CLI interface;
        # this is not an independent mathematical MP3 decoder oracle.
        mp3 = scratch / "reference.mp3"
        cli("-y", "-i", pcm_path, "-c:a", "libmp3lame", "-b:a", "128k", mp3)
        raw = cli("-i", mp3, "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1")
        expected = (
            torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(-1, 2).T
        )
        actual, sr = ta.load(mp3)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        assert sr == rate and actual.device.type == "cpu"
        passed(
            "mp3-decode-versus-ffmpeg-cli",
            max_absolute_error=float((actual - expected).abs().max()),
        )
        # Save different CPU views; verify independently parsed PCM samples,
        # not just a roundtrip through the same decoder API.
        for extension in ("wav", "flac"):
            for layout in ("channels-first-strided", "channels-last", "mono-1d"):
                value = pcm[:, ::2] if layout.endswith("strided") else pcm
                source = (
                    value
                    if layout.endswith("strided")
                    else (value.T if layout == "channels-last" else value[0])
                )
                expected = value if layout != "mono-1d" else value[:1]
                path = scratch / f"save-{layout}.{extension}"
                ta.save(path, source, rate, channels_first=layout != "channels-last")
                decoded_raw = cli(
                    "-i", path, "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1"
                )
                decoded = (
                    torch.frombuffer(bytearray(decoded_raw), dtype=torch.int16)
                    .reshape(-1, expected.shape[0])
                    .T
                )
                torch.testing.assert_close(
                    decoded.float() / 32768, expected, rtol=0, atol=0
                )
                if extension == "wav":
                    with wave.open(str(path), "rb") as stream:
                        assert stream.getnchannels() == expected.shape[0]
                        assert (
                            stream.getframerate() == rate and stream.getsampwidth() == 2
                        )
                        integers = struct.unpack(
                            "<" + "h" * (expected.numel()),
                            stream.readframes(stream.getnframes()),
                        )
                        independent = (
                            torch.tensor(integers, dtype=torch.float32)
                            .reshape(-1, expected.shape[0])
                            .T
                            / 32768
                        )
                        exact(independent, expected)
                passed(f"save-{layout}-{extension}-independent-pcm")
        # Direct encoder binary output accepted by the public torchaudio loader.
        encoded = AudioEncoder(pcm, sample_rate=rate).to_tensor(format="wav")
        actual, sr = ta.load(io.BytesIO(bytes(encoded.tolist())))
        exact(actual, pcm)
        assert sr == rate
        passed("direct-encoder-bytes-public-loader")
        # Existing API ignores normalization=False with an explicit warning.
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            actual, sr = ta.load(pcm_path, normalize=False)
        exact(actual, pcm)
        assert any("normalize=False" in str(w.message) for w in captured)
        passed("public-normalize-false-warning-and-normalized-result")
        for name, function in (
            ("missing-file", lambda: ta.load(scratch / "missing.wav")),
            ("malformed-file", lambda: ta.load(io.BytesIO(b"not a media stream"))),
            ("zero-sample-rate", lambda: ta.save(scratch / "invalid.wav", pcm, 0)),
            (
                "invalid-rank",
                lambda: ta.save(scratch / "invalid.wav", pcm.unsqueeze(0), rate),
            ),
        ):
            try:
                function()
            except (ValueError, RuntimeError):
                passed("refuse-" + name)
            else:
                raise AssertionError("Expected rejection: " + name)
        # CPU file IO followed by native GPU resampling, with output and weighted
        # input-gradient comparison to CPU double. File decoding has no autograd.
        loaded, sr = ta.load(pcm_path)
        reference = loaded.double().requires_grad_()
        expected = ta.functional.resample(reference, sr, 8000)
        weights = torch.linspace(
            -0.75, 0.9, expected.numel(), dtype=torch.float64
        ).reshape(expected.shape)
        expected_grad = torch.autograd.grad((expected * weights).sum(), reference)[0]
        gpu = loaded.cuda().requires_grad_()
        result = ta.functional.resample(gpu, sr, 8000)
        gradient = torch.autograd.grad((result * weights.cuda().float()).sum(), gpu)[0]
        torch.cuda.synchronize()
        assert result.is_cuda and gradient.is_cuda
        torch.testing.assert_close(
            result.cpu().double(), expected, rtol=0.002, atol=3e-5
        )
        torch.testing.assert_close(
            gradient.cpu().double(), expected_grad, rtol=0.002, atol=3e-5
        )
        saved = scratch / "gpu-result-explicit-cpu-transfer.wav"
        ta.save(saved, result.detach().cpu(), 8000)
        readback, sr = ta.load(saved)
        assert sr == 8000
        torch.testing.assert_close(
            readback, result.detach().cpu(), rtol=0, atol=1 / 32768
        )
        passed("cpu-codec-hip-resample-forward-backward-explicit-cpu-save")
        report["fixture_sha256"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in scratch.iterdir()
            if p.is_file()
        }
        from torchcodec._core import get_ffmpeg_library_versions

        report["ffmpeg_library_versions"] = get_ffmpeg_library_versions()
        report["cxx11_abi"] = torch._C._GLIBCXX_USE_CXX11_ABI
        mapped = set()
        for line in Path("/proc/self/maps").read_text().splitlines():
            name = line.split()[-1]
            if name.startswith("/") and any(
                v in name
                for v in ("libtorchcodec", "/libav", "/libswresample", "/libswscale")
            ):
                mapped.add(name)
        report["mapped_codec_sha256"] = {
            n: hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in sorted(mapped)
        }
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        persist()


if __name__ == "__main__":
    main()
