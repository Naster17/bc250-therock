# SPDX-License-Identifier: MIT
"""Supplemental eager-training runtime identity; does not replace run fingerprints."""

import hashlib
import json
import os
from pathlib import Path
import platform
import sys

SCHEMA = 1
FILENAME = "runtime-provenance.json"
MARKER = "runtime_provenance_sha256"
ENVIRONMENT = (
    "BC250_QWEN35_TRAINING_OPS",
    "BC250_GATED_DELTA",
    "BC250_GEMM_TUNING",
    "HSA_ENABLE_SDMA",
    "HSA_XNACK",
    "HSA_OVERRIDE_GFX_VERSION",
    "PYTORCH_ROCM_ARCH",
    "PYTORCH_CUDA_ALLOC_CONF",
    "PYTORCH_ALLOC_CONF",
    "ROCBLAS_USE_HIPBLASLT",
    "TORCH_BLAS_PREFER_HIPBLASLT",
    "CUBLAS_WORKSPACE_CONFIG",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
)


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def capture(torch, trainer, fingerprint, device):
    import transformers

    trainer = Path(trainer)
    files = {}

    def include(key, path, required=True):
        path = Path(path)
        if path.is_file():
            files[key] = file_sha(path)
        elif required:
            raise ValueError(f"Cannot identify runtime file: {key}")

    for name in (
        "train.py",
        "formatting.py",
        "bc250_ops.py",
        "bc250_gated_delta.py",
        Path(__file__).name,
    ):
        include("application/" + name, trainer.parent / name)
    torchroot = Path(torch.__file__).parent
    include("torch/_C", torch._C.__file__)
    for name in ("libtorch_cpu.so", "libtorch_python.so", "libc10.so"):
        include("torch/lib/" + name, torchroot / "lib" / name)
    for name in ("nn/bc250_training_ops.py", "nn/attention/_gfx1013.py"):
        include("torch/" + name, torchroot / name)
    include(
        "transformers/qwen3_5",
        Path(transformers.__file__).parent / "models/qwen3_5/modeling_qwen3_5.py",
    )
    if device == "cuda":
        for name in ("libtorch_hip.so", "libc10_hip.so"):
            include("torch/lib/" + name, torchroot / "lib" / name)
        rocm = Path(os.environ.get("ROCM_PATH", "/workspace/build/dist/rocm"))
        for name in (
            "libamdhip64.so",
            "libhsa-runtime64.so",
            "librocblas.so",
            "libhipblas.so",
            "libhipblaslt.so",
            "libMIOpen.so",
        ):
            include("rocm/" + name, rocm / "lib" / name)
        library = Path(
            os.environ.get("ROCBLAS_TENSILE_LIBPATH", str(rocm / "lib/rocblas/library"))
        )
        assets = sorted(
            p
            for p in library.rglob("*")
            if p.is_file() and "gfx1013" in str(p.relative_to(library))
        )
        if not assets:
            raise ValueError("No gfx1013 rocBLAS assets available for provenance")
        for path in assets:
            include("rocblas-assets/" + str(path.relative_to(library)), path)
    # Hash actual mapped core DSOs as well as the selected ROCm files, so loader
    # search overrides of already loaded HIP/ATen libraries are visible.
    prefixes = (
        "libtorch",
        "libc10",
        "libamdhip64",
        "libhsa-runtime64",
        "librocblas",
        "libhipblas",
        "libMIOpen",
        "libstdc++",
        "libc.",
    )
    mapped = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            path = Path(fields[5])
            if path.name.startswith(prefixes):
                value = file_sha(path)
                if path.name in mapped and mapped[path.name] != value:
                    raise ValueError("Ambiguous mapped native library: " + path.name)
                mapped[path.name] = value
    preset = os.environ.get("ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH")
    preload = os.environ.get("LD_PRELOAD", "")
    if preload:
        raise ValueError(
            "Checkpoint provenance does not support LD_PRELOAD; use the qualified launcher"
        )
    identity = dict(
        files=files,
        mapped_native=mapped,
        preset_sha256=file_sha(preset) if preset else None,
        environment={key: os.environ.get(key) for key in ENVIRONMENT},
        python=platform.python_version(),
        kernel=platform.release(),
        torch=str(torch.__version__),
        hip=torch.version.hip,
        architecture=(
            torch.cuda.get_device_properties(0).gcnArchName
            if device == "cuda"
            else "cpu"
        ),
        device=device,
    )
    return dict(schema=SCHEMA, fingerprint=fingerprint, identity=identity)


def read_checkpoint(checkpoint, run, fingerprint):
    path = Path(checkpoint) / FILENAME
    marker = run.get(MARKER)
    if marker is None:
        if MARKER in run or path.exists():
            raise ValueError("Orphan or invalid checkpoint runtime provenance")
        return None
    if not isinstance(marker, str) or len(marker) != 64:
        raise ValueError("Invalid checkpoint runtime provenance digest")
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(
            "Missing or malformed checkpoint runtime provenance"
        ) from error
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "fingerprint", "identity"}
        or type(value["schema"]) is not int
        or value["schema"] != SCHEMA
        or value["fingerprint"] != fingerprint
        or not isinstance(value["identity"], dict)
        or digest(value) != marker
    ):
        raise ValueError("Invalid or corrupted checkpoint runtime provenance")
    return value


def check_resume(expected, current):
    if expected is None:
        print(
            "[bc250] Legacy checkpoint has no runtime provenance; existing resume checks apply, numerical-runtime continuity is unverified.",
            file=sys.stderr,
            flush=True,
        )
    elif expected != current:
        old, new = expected["identity"], current["identity"]
        changed = sorted(
            key for key in old.keys() | new.keys() if old.get(key) != new.get(key)
        )
        raise ValueError(
            "Checkpoint numerical runtime mismatch ("
            + ", ".join(changed)
            + "); restore the checkpoint runtime/preset/hooks. No output written."
        )


def write_checkpoint(destination, record):
    (Path(destination) / FILENAME).write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    return digest(record)
