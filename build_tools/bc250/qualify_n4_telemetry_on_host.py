#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""N4 read-only telemetry provenance + semantics audit on the current r2 release.

Adapts the proven kernel/r3/telemetry/sample-on-host.py methodology to the D61 r2
release: discover the running SMU governor, read its retained original sysfs
metrics descriptor (bypassing the tmpfs overlay WITHOUT unmounting), sample
whitelisted clock/temperature/power/fan sensors plus read-only DRM info ioctls
through idle + one bounded eight-second native integer GPU load + cooldown, and
assert kernel log / mounts / governor-config-fan-binary hashes / power policy are
unchanged. Never changes clocks, voltages, fan policy or boot; never issues raw SMU
mailbox commands or runs the governor control API. Caller-side preflight, a fresh
recovery guard and the exclusive runtime lock wrap the sampling.
"""
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from bc250_torch import runtime_lock
from deploy_torch_a04_on_host import atomic_json
from qualify_training_n33_on_host import WS
from qualify_training_n39_on_host import preflight

KWS = Path("/home/nik/bc250work/kernel-r3-7.2.10")
LABEL = "n4-telemetry-r2"
GUARD = Path("/run") / LABEL
BOOT_GUARD = KWS / "helpers/boot-guard.py"
R2_KERNEL_SOURCE = "6f57bb40d26d1cbec862807fc89a9c61af1b9a39"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return Path(p).read_text().strip()


def main():
    preflight(LABEL)
    logs = KWS / "logs/telemetry-n4-r2"
    logs.mkdir()
    gpu = Path("/sys/class/drm/card0/device").resolve()

    def mounts():
        return [
            l
            for l in Path("/proc/self/mountinfo").read_text().splitlines()
            if str(gpu) in l
        ]

    gov = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            if (proc / "comm").read_text().strip().startswith("cyan-skillfish-"):
                cmd = (proc / "cmdline").read_bytes()
                if b"/usr/local/bin/cyan-skillfish-governor-smu\0" in cmd:
                    gov.append(proc)
        except (OSError, PermissionError):
            pass
    assert len(gov) == 1, gov
    gov = gov[0]
    raw_paths = []
    patched_path = gpu / "gpu_metrics"
    for fd in (gov / "fd").iterdir():
        try:
            if (
                os.readlink(fd) == str(patched_path)
                and fd.stat().st_dev != patched_path.stat().st_dev
            ):
                raw_paths.append(fd)
        except OSError:
            pass
    assert (
        len(raw_paths) == 1
    ), "Require a verified original sysfs descriptor; never unmount the overlay"
    raw_path = raw_paths[0]
    hw = {read(p / "name"): p for p in Path("/sys/class/hwmon").glob("hwmon*")}
    assert {"amdgpu", "k10temp", "nct6686"} <= hw.keys()
    config = Path("/etc/cyan-skillfish-governor-smu/config.toml")
    report = {
        "status": "running",
        "boot_id": read("/proc/sys/kernel/random/boot_id"),
        "kernel": os.uname().release,
        "kernel_source_import": R2_KERNEL_SOURCE,
        "amdgpu": "built-in (r2); not a loadable module",
        "governor_pid": int(gov.name),
        "governor_binary_sha256": sha(gov / "exe"),
        "governor_config_sha256": sha(config),
        "fan_script_sha256": sha("/usr/local/bin/bc250-fan-curve"),
        "mounts_before": mounts(),
        "hwmon_paths": {n: str(p.resolve()) for n, p in hw.items()},
        "raw_kernel_metrics_fd": str(raw_path),
        "raw_device": raw_path.stat().st_dev,
        "overlay_device": patched_path.stat().st_dev,
        "probe_sha256": sha(KWS / "telemetry-reference-load"),
        "kernel_power_policy": read(gpu / "power_dpm_force_performance_level"),
        "sampling_note": "Sequential non-atomic readings; governor remains active and may change policy outputs normally. Original descriptor gives kernel-translated v2.2 metrics, not raw PMFW bytes.",
        "limits": "One eight-second integer workload; no physical clock calibration, thermal stress/accuracy certification or legacy hybrid-layout validation; no raw SMU mailbox commands",
    }
    assert report["raw_device"] != report["overlay_device"]

    def metrics(path):
        b = Path(path).read_bytes()
        assert len(b) == 128 and b[:4] == bytes([128, 0, 2, 2]), (
            str(path),
            len(b),
            b[:4].hex(),
        )
        u16 = lambda o: int.from_bytes(b[o : o + 2], "little")
        return dict(
            gfx_centi_c=u16(4),
            soc_centi_c=u16(6),
            core_centi_c=[u16(8 + 2 * i) for i in range(8)],
            gfx_activity_centi_percent=u16(28),
            timestamp_ns=int.from_bytes(b[32:40], "little"),
            average_gfx_mhz=u16(64),
            current_gfx_mhz=u16(76),
            current_soc_mhz=u16(78),
            core_mhz=[u16(88 + 2 * i) for i in range(8)],
            hex=b.hex(),
        )

    class Info(ctypes.Structure):
        _fields_ = [
            ("ptr", ctypes.c_uint64),
            ("size", ctypes.c_uint32),
            ("query", ctypes.c_uint32),
            ("params", ctypes.c_uint64 * 2),
        ]

    assert ctypes.sizeof(Info) == 32
    render = os.open("/dev/dri/renderD128", os.O_RDONLY | os.O_CLOEXEC)

    def sensor(kind):
        value = ctypes.c_uint32()
        args = Info(ctypes.addressof(value), 4, 0x1D, (kind, 0))
        try:
            fcntl.ioctl(render, 0x40206445, bytearray(bytes(args)), True)
            return {"value": value.value}
        except OSError as e:
            return {"errno": e.errno}

    release = Path("/opt/bc250-stack/current").resolve()
    manifest = json.loads((release / "runtime.json").read_text())
    report["runtime_image"] = manifest["image"]
    cmd = [
        "podman",
        "run",
        "--rm",
        "--name",
        "bc250-n4-telemetry-audit",
        "--network",
        "none",
        "--device",
        "/dev/kfd",
        "--device",
        "/dev/dri/renderD128",
        "--user",
        f"{manifest['uid']}:{manifest['gid']}",
    ]
    for gid in sorted({os.stat(p).st_gid for p in ("/dev/kfd", "/dev/dri/renderD128")}):
        cmd += ["--group-add", str(gid)]
    cmd += [
        "-v",
        f'{release / "rocm"}:/opt/rocm:ro',
        "-v",
        f"{KWS}:/kernel:ro",
        "-e",
        "HSA_ENABLE_SDMA=0",
        "-e",
        "LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/llvm/lib",
        manifest["image"],
        "timeout",
        "--kill-after=3s",
        "20s",
        "/kernel/telemetry-reference-load",
    ]
    guard = subprocess.Popen(
        [
            "python3",
            str(BOOT_GUARD),
            "--release",
            os.uname().release,
            "--phase",
            LABEL,
            "--timeout",
            "600",
            "--directory",
            str(GUARD),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.monotonic() + 60
    while not (GUARD / "state.json").exists():
        assert (
            time.monotonic() < deadline and guard.poll() is None
        ), "boot guard did not arm"
        time.sleep(0.05)
    before = subprocess.check_output(["dmesg"], text=True)
    (logs / "kernel-before.log").write_text(before)
    process = None
    last_exit = None
    count = 0
    try:
        with runtime_lock(
            Path("/opt/bc250-stack/current/.runtime.lock"), exclusive=True
        ):
            with (logs / "samples.jsonl").open("w") as output, (
                logs / "gpu-load.log"
            ).open("w") as gpu_log:
                start = time.monotonic()
                idle_end = start + 4
                cool_end = None
                load_start = None
                while True:
                    now = time.monotonic()
                    if process is None and now >= idle_end:
                        process = subprocess.Popen(
                            cmd, stdout=gpu_log, stderr=subprocess.STDOUT
                        )
                        load_start = now
                    if (
                        process is not None
                        and last_exit is None
                        and process.poll() is not None
                    ):
                        last_exit = process.returncode
                        cool_end = now + 4
                        assert last_exit == 0, "GPU reference process failed"
                    if cool_end is not None and now >= cool_end:
                        break
                    if (
                        load_start is not None
                        and now - load_start > 25
                        and last_exit is None
                    ):
                        raise RuntimeError("GPU load exceeded bound")
                    phase = (
                        "idle"
                        if process is None
                        else (
                            "post-load"
                            if last_exit is not None
                            else "load-process-active"
                        )
                    )
                    tick = time.monotonic()
                    s = {
                        "seconds": now - start,
                        "phase": phase,
                        "kernel_metrics": metrics(raw_path),
                        "patched_metrics": metrics(patched_path),
                        "pp_dpm_sclk": read(gpu / "pp_dpm_sclk"),
                        "hwmon_freq_hz": int(read(hw["amdgpu"] / "freq1_input")),
                        "gpu_edge_milli_c": int(read(hw["amdgpu"] / "temp1_input")),
                        "cpu_tctl_milli_c": int(read(hw["k10temp"] / "temp1_input")),
                        "nct_cpu_milli_c": int(read(hw["nct6686"] / "temp1_input")),
                        "fan_rpm": int(read(hw["nct6686"] / "fan2_input")),
                        "fan_pwm": int(read(hw["nct6686"] / "pwm2")),
                        "fan_enable": read(hw["nct6686"] / "pwm2_enable"),
                        "drm_gfx_mhz": sensor(1),
                        "drm_gpu_temp_milli_c": sensor(3),
                        "drm_gpu_load_percent": sensor(4),
                    }
                    s["read_seconds"] = time.monotonic() - tick
                    output.write(json.dumps(s) + "\n")
                    output.flush()
                    count += 1
                    assert (
                        max(s["cpu_tctl_milli_c"], s["gpu_edge_milli_c"]) < 80000
                    ), "Temperature bound exceeded"
                    time.sleep(max(0, 0.2 - (time.monotonic() - tick)))
            assert (
                "TELEMETRY_GPU_PASS checks=131072"
                in (logs / "gpu-load.log").read_text()
            )
            after = subprocess.check_output(["dmesg"], text=True)
            (logs / "kernel-after.log").write_text(after)
            report["kernel_log_identical"] = before == after
            assert before == after, "Kernel log changed; inspect evidence"
            report["mounts_after"] = mounts()
            assert report["mounts_before"] == report["mounts_after"]
            assert sha(config) == report["governor_config_sha256"]
            assert sha("/usr/local/bin/bc250-fan-curve") == report["fan_script_sha256"]
            assert sha(gov / "exe") == report["governor_binary_sha256"]
            assert (
                read(gpu / "power_dpm_force_performance_level")
                == report["kernel_power_policy"]
            )
            report.update(status="passed", samples=count, process_exit=last_exit)
    except BaseException as error:
        report.update(status="failed", error=repr(error), samples=count)
        raise
    finally:
        if process is not None and process.poll() is None:
            subprocess.run(
                ["podman", "stop", "--time", "2", "bc250-n4-telemetry-audit"],
                timeout=10,
                check=False,
            )
        os.close(render)
        (logs / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        # The boot guard acknowledges only ack.json carrying its exact
        # boot_id/nonce/phase; an unacknowledged deadline triggers emergency
        # restart, so this must match the guard's nonce, not a plain marker.
        state = json.loads((GUARD / "state.json").read_text())
        (GUARD / "ack.json").write_text(
            json.dumps({k: state[k] for k in ("boot_id", "nonce", "phase")})
        )
        guard.wait(timeout=15)
        state = json.loads((GUARD / "state.json").read_text())
        assert state["status"] == "acknowledged", state
        atomic_json(logs / "final.json", report)
    assert report["status"] == "passed"
    print("N4_TELEMETRY_READONLY_PASS", count, flush=True)


if __name__ == "__main__":
    main()
