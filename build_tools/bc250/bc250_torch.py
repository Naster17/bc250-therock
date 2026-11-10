#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run Python or a shell with persistent host paths in the BC-250 Torch release."""

import argparse
from contextlib import contextmanager
from collections.abc import Iterator
import fcntl
import hashlib
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import pwd
import re
import signal
import stat
import subprocess
import sys
import tempfile


@dataclass
class LaunchOptions:
    arguments: list[str] = field(default_factory=list)
    mounts: list[tuple[str, str]] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)
    shm_size: str = "1g"
    shell: bool = False
    runtime_profile: str = "daily"


RCCL_SINGLE_ENVIRONMENT = {
    "NCCL_WIN_ENABLE": "0",
    "NCCL_CUMEM_ENABLE": "0",
    "NCCL_CUMEM_HOST_ENABLE": "0",
    "NCCL_P2P_DISABLE": "1",
    "NCCL_IB_DISABLE": "1",
    "NCCL_SOCKET_IFNAME": "lo",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
}


PROTECTED_ENVIRONMENT = {
    "HOME",
    "PATH",
    "LD_LIBRARY_PATH",
    "ROCM_PATH",
    "HIP_PATH",
    "HSA_ENABLE_SDMA",
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "PYTHONHOME",
    "USER",
    "LOGNAME",
    "PYTHONDONTWRITEBYTECODE",
}
INHERITED_PREFIXES = ("HF_", "HUGGINGFACE_", "WANDB_", "MLFLOW_")
INHERITED_NAMES = {
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "TORCHINDUCTOR_COMPILE_THREADS",
    "XDG_CACHE_HOME",
    "TORCH_HOME",
    "PYTHONFAULTHANDLER",
    "BC250_GEMM_TUNING",
}
PROTECTED_PATHS = (
    Path("/workspace"),
    Path("/opt/torch-venv"),
    Path("/opt/bc250-tools"),
    Path("/usr"),
    Path("/bin"),
    Path("/lib"),
    Path("/lib64"),
    Path("/etc"),
    Path("/dev"),
    Path("/proc"),
    Path("/sys"),
)


def check_environment(name: str, value: str) -> None:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"Invalid environment variable name: {name}")
    if name in PROTECTED_ENVIRONMENT:
        raise ValueError(f"The validated runtime controls {name}")
    if "\n" in value or "\r" in value or "\0" in value:
        raise ValueError(f"Environment variable {name} must be a single line")


def check_mount(path: Path) -> None:
    if not path.is_absolute() or not path.exists():
        raise ValueError(f"Mount source must exist: {path}")
    if ":" in str(path) or "\n" in str(path):
        raise ValueError(f"Unsupported mount path character: {path}")
    if any(
        path == protected
        or protected.is_relative_to(path)
        or path.is_relative_to(protected)
        for protected in PROTECTED_PATHS
    ):
        raise ValueError(f"Mount would hide container runtime files: {path}")


def launcher_help() -> None:
    parser = argparse.ArgumentParser(
        prog="bc250-torch",
        description=__doc__,
        usage="%(prog)s [launcher options] [--] [python options] script.py [script arguments]",
        epilog="Options must precede Python arguments. Host paths retain their names; home, cwd and script parent are writable. -h/--help go to Python.",
        add_help=False,
    )
    parser.add_argument(
        "--launcher-help", action="store_true", help="show this launcher help"
    )
    parser.add_argument(
        "--runtime-profile",
        choices=("daily", "rccl-single"),
        help="explicit opt-in RCCL rank-one profile; default: daily",
    )
    parser.add_argument(
        "--mount",
        metavar="PATH",
        action="append",
        help="mount an existing host file/directory read-write at the same path",
    )
    parser.add_argument(
        "--mount-ro",
        metavar="PATH",
        action="append",
        help="mount an existing host file/directory read-only at the same path",
    )
    parser.add_argument(
        "--env",
        metavar="NAME[=VALUE]",
        action="append",
        help="pass a variable from the caller or set its value",
    )
    parser.add_argument(
        "--env-file",
        metavar="PATH",
        action="append",
        help="read NAME=VALUE lines; blank lines and # comments allowed",
    )
    parser.add_argument(
        "--shm-size",
        metavar="SIZE",
        help="DataLoader shared-memory limit (default: 1g; e.g. 2g)",
    )
    parser.add_argument(
        "--shell",
        action="store_true",
        help="run bash; optional remaining arguments go to bash",
    )
    parser.print_help()


def parse_launcher_arguments(
    arguments: list[str], environment: dict[str, str]
) -> LaunchOptions:
    options = LaunchOptions(
        environment={
            name: value
            for name, value in environment.items()
            if name in INHERITED_NAMES or name.startswith(INHERITED_PREFIXES)
        }
    )
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--launcher-help":
            launcher_help()
            raise SystemExit(0)
        if argument == "--":
            index += 1
            break
        if argument == "--shell":
            options.shell = True
            index += 1
            break
        if argument not in (
            "--mount",
            "--mount-ro",
            "--env",
            "--env-file",
            "--shm-size",
            "--runtime-profile",
        ):
            break
        if index + 1 >= len(arguments):
            raise ValueError(f"Missing value for {argument}")
        value = arguments[index + 1]
        if argument == "--runtime-profile":
            options.runtime_profile = value
        elif argument.startswith("--mount"):
            path = Path(value).expanduser().absolute()
            check_mount(path)
            options.mounts.append(
                (str(path), "ro" if argument == "--mount-ro" else "rw")
            )
        elif argument == "--env":
            name, separator, contents = value.partition("=")
            if not separator:
                if name not in environment:
                    raise ValueError(f"Caller environment variable is unset: {name}")
                contents = environment[name]
            options.environment[name] = contents
        elif argument == "--env-file":
            for line in Path(value).expanduser().read_text().splitlines():
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                name, separator, contents = line.partition("=")
                if not separator:
                    raise ValueError(
                        f"Expected NAME=VALUE in environment file: {value}"
                    )
                options.environment[name.strip()] = contents
        else:
            options.shm_size = value
        index += 2
    options.arguments = arguments[index:]
    validate_options(options)
    return options


def validate_options(options: LaunchOptions) -> None:
    if options.runtime_profile not in ("daily", "rccl-single"):
        raise ValueError("Unknown runtime profile")
    if not re.fullmatch(r"[1-9][0-9]*[bkmgBKMG]?", options.shm_size):
        raise ValueError("Shared memory size must be positive, e.g. 1g or 512m")
    for name, value in options.environment.items():
        check_environment(name, value)
    # D12 retained an import-time ROCr SIGSEGV with this HIP sentinel. Refuse
    # that specific input before elevation/container startup, including contexts.
    # Do not reinterpret ROCR filters, UUIDs, CUDA aliases or trailing terminators.
    if options.environment.get("HIP_VISIBLE_DEVICES") == "-1":
        raise ValueError(
            "HIP_VISIBLE_DEVICES=-1 is refused by this BC-250 launcher after a "
            "recorded ROCr import crash. Unset HIP_VISIBLE_DEVICES for CPU-only "
            "Python work and avoid torch.cuda initialization; use 0 to select "
            "this board's GPU. This does not disable GPU access."
        )
    for source, mode in options.mounts:
        if mode not in ("rw", "ro"):
            raise ValueError(f"Invalid mount mode: {mode}")
        check_mount(Path(source))


def selected_release(options: LaunchOptions) -> Path:
    path = (
        Path("/opt/bc250-stack/current")
        if options.runtime_profile == "daily"
        else Path("/opt/bc250-stack/profiles/rccl-single")
    )
    return path.resolve(strict=True)


def qualified_rccl_profile(
    release: Path, manifest: dict, options: LaunchOptions
) -> bool:
    """Fail closed on mismatched native payloads or broadened opt-in settings."""
    config = manifest.get("rccl_single_profile")
    if not config:
        if options.runtime_profile == "rccl-single":
            raise ValueError("Selected release has no qualified RCCL profile")
        return False
    if options.runtime_profile != "rccl-single":
        raise ValueError("RCCL profile requires explicit --runtime-profile rccl-single")
    if config["environment"] != RCCL_SINGLE_ENVIRONMENT:
        raise ValueError("RCCL profile settings differ from the qualified subset")
    for name, value in RCCL_SINGLE_ENVIRONMENT.items():
        if name in options.environment and options.environment[name] != value:
            raise ValueError(f"The qualified RCCL profile controls {name}")
    identities = config["identity_sha256"]
    required = {"torch/torch/lib/libtorch_hip.so", "rccl/lib/librccl.so.1.0"}
    if not required.issubset(identities):
        raise ValueError("Missing RCCL/Torch profile identities")
    for name, digest in identities.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid RCCL profile identity path")
        with (release / relative).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError(f"RCCL profile identity mismatch: {name}")
    library = release / "rccl/lib/librccl.so.1.0"
    for name in ("librccl.so", "librccl.so.1"):
        if (library.parent / name).resolve(strict=True) != library.resolve(strict=True):
            raise ValueError("RCCL profile SONAME link mismatch")
    return True


def prepare_python_arguments(
    arguments: list[str], workspace: Path
) -> tuple[list[str], list[tuple[Path, str]]]:
    """Preserve absolute host script paths and mount external script parents."""
    arguments = arguments.copy()
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument in ("-c", "-m", "-", "-h", "--help", "-V", "--version"):
            return arguments, []
        if argument == "--":
            index += 1
            break
        if argument in ("-W", "-X", "--check-hash-based-pycs"):
            index += 2
        elif argument.startswith("-"):
            index += 1
        else:
            break
    if index >= len(arguments):
        return arguments, []
    script = Path(arguments[index]).expanduser()
    if not script.is_absolute():
        script = workspace / script
    # Keep the lexical name so __file__ and symlinks retain host semantics.
    script = script.absolute()
    if not script.is_file():
        raise FileNotFoundError(f"Python script is not a file: {script}")
    arguments[index] = str(script)
    directories = {script.parent, script.resolve(strict=True).parent}
    return arguments, [
        (directory, str(directory))
        for directory in sorted(directories)
        if not directory.is_relative_to(workspace)
    ]


def qualified_gemm_override(
    release: Path, manifest: dict, options: LaunchOptions
) -> Path | None:
    config = manifest.get("rocblas_gemm_tuning")
    if (
        not config
        or not config.get("enabled", False)
        or options.environment.get("BC250_GEMM_TUNING") == "0"
        or "ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH" in options.environment
    ):
        return None
    path = release / "rocblas-gemm-n32.csv"
    try:
        identities = config["identity_sha256"]
        if len(identities) < 2:
            raise ValueError("Missing tuning identities")
        files = {path: config["sha256"]}
        for name, digest in identities.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Invalid tuning identity path")
            files[release / "rocm" / relative] = digest
        for source, digest in files.items():
            with source.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                    raise ValueError("Tuning identity mismatch")
    except (OSError, KeyError, TypeError, ValueError):
        print(
            "bc250-torch: GEMM tuning skipped because preset identities differ",
            file=sys.stderr,
        )
        return None
    return path


def build_command(
    release: Path,
    manifest: dict,
    workspace: Path,
    home: Path,
    options: LaunchOptions,
    groups: list[int],
    environment_file: Path,
) -> list[str]:
    rccl_profile = qualified_rccl_profile(release, manifest, options)
    arguments, script_mounts = (
        (options.arguments, [])
        if options.shell
        else prepare_python_arguments(options.arguments, workspace)
    )
    mounts = {str(home): "rw", str(workspace): "rw"}
    mounts.update({destination: "rw" for _, destination in script_mounts})
    explicit_mounts = dict(options.mounts)
    # Automatic child mounts must not undo an explicit read-only parent.
    for automatic in mounts:
        ancestors = [
            (Path(source), mode)
            for source, mode in explicit_mounts.items()
            if Path(automatic).is_relative_to(Path(source))
        ]
        if ancestors:
            mounts[automatic] = max(ancestors, key=lambda item: len(item[0].parts))[1]
    mounts.update(explicit_mounts)
    command = [
        "podman",
        "run",
        "--rm",
        "--pull=never",
        "--network",
        "host",
        "-i",
        "--shm-size",
        options.shm_size,
        "--user",
        f"{manifest['uid']}:{manifest['gid']}",
    ]
    if sys.stdin.isatty() and sys.stdout.isatty():
        command.append("-t")
    for group in sorted(set(groups)):
        command += ["--group-add", str(group)]
    for device in ("/dev/kfd", "/dev/dri/renderD128"):
        command += ["--device", device]
    for source, mode in mounts.items():
        check_mount(Path(source))
        command += ["-v", f"{source}:{source}:{mode}"]
    for source, destination, mode in (
        (release / "rocm", "/workspace/build/dist/rocm", "ro"),
        (release / "torch", "/workspace/pytorch-work", "ro"),
        (release / "cache", "/workspace/user-cache", "rw"),
        (Path("/etc/passwd"), "/etc/passwd", "ro"),
        (Path("/etc/group"), "/etc/group", "ro"),
    ):
        command += ["-v", f"{source}:{destination}:{mode}"]
    tuning = qualified_gemm_override(release, manifest, options)
    if tuning is not None:
        check_mount(tuning)
        command += ["-v", f"{tuning}:/workspace/rocblas-gemm-overrides.csv:ro"]
        command += [
            "-e",
            "ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH=/workspace/rocblas-gemm-overrides.csv",
        ]
    command += ["-w", str(workspace), "--env-file", str(environment_file)]
    fixed = {
        "HOME": str(home),
        "USER": pwd.getpwuid(manifest["uid"]).pw_name,
        "LOGNAME": pwd.getpwuid(manifest["uid"]).pw_name,
        "ROCM_PATH": "/workspace/build/dist/rocm",
        "HIP_PATH": "/workspace/build/dist/rocm",
        "LD_LIBRARY_PATH": "/workspace/build/dist/rocm/lib:/workspace/build/dist/rocm/lib/llvm/lib",
        "PATH": "/workspace/build/dist/rocm/bin:/workspace/build/dist/rocm/lib/llvm/bin:/opt/torch-venv/bin:/usr/local/therock-tools/bin:/usr/bin:/bin",
        "HSA_ENABLE_SDMA": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TRITON_CACHE_DIR": "/workspace/user-cache/triton",
        "TORCHINDUCTOR_CACHE_DIR": "/workspace/user-cache/inductor",
    }
    if rccl_profile:
        command += ["-v", f"{release / 'rccl'}:/workspace/rccl:ro"]
        fixed["LD_LIBRARY_PATH"] = "/workspace/rccl/lib:" + fixed["LD_LIBRARY_PATH"]
        fixed.update(RCCL_SINGLE_ENVIRONMENT)
    if "TORCHINDUCTOR_COMPILE_THREADS" not in options.environment:
        fixed["TORCHINDUCTOR_COMPILE_THREADS"] = "4"
    if "PYTHONFAULTHANDLER" not in options.environment:
        fixed["PYTHONFAULTHANDLER"] = "1"
    for name, value in fixed.items():
        command += ["-e", f"{name}={value}"]
    interpreter = ["bash", "--noprofile", "--norc"] if options.shell else ["python"]
    return command + [manifest["image"], *interpreter, *arguments]


def read_context(path: Path, uid: int) -> tuple[LaunchOptions, Path]:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor) as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != uid
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise ValueError(
                "Invalid private launcher context ownership or permissions"
            )
        context = json.load(stream)
    options = LaunchOptions(**context["options"])
    validate_options(options)
    return options, Path(context["workspace"])


def exit_status(status: int) -> int:
    return status if status >= 0 else 128 - status


def run_process(command: list[str]) -> int:
    """Keep cleanup active while forwarding termination to the runtime."""
    with subprocess.Popen(command) as process:
        previous = signal.getsignal(signal.SIGTERM)

        def terminate(signum: int, frame: object) -> None:
            if process.poll() is None:
                process.send_signal(signum)

        if os.geteuid() == 0:
            signal.signal(signal.SIGTERM, terminate)
        try:
            return process.wait()
        except KeyboardInterrupt:
            if process.poll() is None:
                try:
                    process.send_signal(signal.SIGINT)
                except PermissionError:
                    # The terminal already signals the elevated foreground job.
                    pass
            process.wait()
            return 130
        finally:
            if os.geteuid() == 0:
                signal.signal(signal.SIGTERM, previous)


@contextmanager
def runtime_lock(path: Path, *, exclusive: bool = False) -> Iterator[None]:
    """Serialize developer-prefix updates against launcher processes.

    Every updater must cooperate; this cannot protect arbitrary manual writes.
    """
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
            raise ValueError("Invalid runtime lock ownership or file type")
        mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(descriptor, mode | fcntl.LOCK_NB)
        except BlockingIOError:
            print(
                "bc250-torch: waiting for the developer runtime lock", file=sys.stderr
            )
            fcntl.flock(descriptor, mode)
        yield
    finally:
        os.close(descriptor)


def main() -> None:
    arguments = sys.argv[1:]
    if "--launcher-help" in arguments[:1]:
        launcher_help()
        return
    daily = Path("/opt/bc250-stack/current").resolve(strict=True)
    daily_manifest = json.loads((daily / "runtime.json").read_text())
    if arguments[:1] == ["--_context"]:
        if os.geteuid() != 0 or len(arguments) != 2:
            raise ValueError(
                "Private launcher contexts are only accepted after elevation"
            )
        options, workspace = read_context(Path(arguments[1]), daily_manifest["uid"])
    else:
        workspace = Path.cwd().resolve()
        options = parse_launcher_arguments(arguments, dict(os.environ))
        if not options.shell:
            prepare_python_arguments(options.arguments, workspace)
    release = selected_release(options)
    manifest = json.loads((release / "runtime.json").read_text())
    if not manifest["validated"]:
        raise RuntimeError("The selected BC-250 release has not passed validation")
    if (manifest["uid"], manifest["gid"]) != (
        daily_manifest["uid"],
        daily_manifest["gid"],
    ):
        raise ValueError("Runtime profile belongs to a different host user")
    if os.geteuid() != 0:
        if os.geteuid() != manifest["uid"]:
            raise RuntimeError("This release belongs to a different host user")
        with tempfile.TemporaryDirectory(prefix="bc250-torch-context-") as directory:
            path = Path(directory) / "context.json"
            path.touch(mode=0o600)
            path.write_text(
                json.dumps({"workspace": str(workspace), "options": vars(options)})
            )
            status = run_process(
                ["doas", str(Path(__file__).resolve()), "--_context", str(path)]
            )
        raise SystemExit(exit_status(status))
    account = pwd.getpwuid(manifest["uid"])
    home = Path(account.pw_dir)
    groups = os.getgrouplist(account.pw_name, manifest["gid"])
    groups += [
        Path(device).stat().st_gid for device in ("/dev/kfd", "/dev/dri/renderD128")
    ]
    with tempfile.TemporaryDirectory(prefix="bc250-torch-env-") as directory:
        path = Path(directory) / "environment"
        path.touch(mode=0o600)
        path.write_text(
            "".join(f"{name}={value}\n" for name, value in options.environment.items())
        )
        # Profile dependencies share the daily mutable ROCm prefix. Acquire its
        # lock before identity checks and keep it until the workload exits.
        if daily_manifest.get("mutable", False):
            with runtime_lock(daily / ".runtime.lock"):
                command = build_command(
                    release, manifest, workspace, home, options, groups, path
                )
                raise SystemExit(exit_status(run_process(command)))
        command = build_command(
            release, manifest, workspace, home, options, groups, path
        )
        raise SystemExit(exit_status(run_process(command)))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, PermissionError) as error:
        print(f"bc250-torch: {error}", file=sys.stderr)
        raise SystemExit(2)
    except KeyboardInterrupt:
        raise SystemExit(130)
