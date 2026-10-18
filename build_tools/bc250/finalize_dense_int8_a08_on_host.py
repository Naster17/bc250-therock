#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Seal tested Lt artifacts and source identities under the exclusive runtime lock."""
import json
import os
from pathlib import Path
import subprocess
from deploy_torch_a04_on_host import atomic_json, check_gate, sha

WS = Path("/home/nik/bc250work/rocm-10.2-gfx1013")
RELEASE = Path("/opt/bc250-stack/current").resolve()


def main() -> None:
    assert os.geteuid() == 0
    assert not subprocess.check_output(["podman", "ps", "-q"], text=True).strip()
    source = json.loads((WS / "helpers/dense-int8-a08-source.json").read_text())
    q = json.loads((WS / "logs/dense-int8-a08-qualification/result.json").read_text())
    assert q["status"] == "passed" and len(q["passed_gates"]) == 10
    labels = q["passed_gates"] + ["dense-int8-a08-benchmark"]
    for label in labels:
        check_gate(label)
    assert (
        json.loads((WS / "logs/dense-int8-a08-benchmark/result.json").read_text())[
            "status"
        ]
        == "passed"
    )
    after = json.loads((WS / "logs/dense-int8-a08-build-r2/result.json").read_text())[
        "after"
    ]
    for p, h in after.items():
        assert sha(Path(p)) == h, p
    for name, h in source["helper_sha256"].items():
        assert sha(WS / "helpers" / name) == h, name
    assert (
        after[
            str(
                WS
                / "source/rocm-libraries/projects/hipblaslt/library/src/amd_detail/gfx1013_scalar.cpp"
            )
        ]
        == source["implementation_sha256"]
    )
    mutable = {
        p.resolve()
        for p in [
            WS / "build/dist/rocm/lib/libhipblaslt.so.1.5",
            WS / "build/dist/rocm/.kpack/blas_lib_gfx1013.kpack",
        ]
    }
    prior = json.loads((WS / "SOURCE_PROFILER_A09.json").read_text())
    protected = {
        p: h
        for p, h in prior["protected_before"].items()
        if Path(p).resolve() not in mutable
    }
    torch = (RELEASE / "torch").resolve()
    for n, h in prior["source"]["candidate_library_sha256"].items():
        protected[str(torch / "torch/lib" / n)] = h
    for n, h in prior["source"]["python_extension_sha256"].items():
        protected[str(torch / n)] = h
    for n, h in prior["source"]["python_sha256"].items():
        protected[str(torch / n)] = h
    assert (torch / "GIT_REVISION").read_text().strip() == source["torch_revision"]
    protected[str(torch / "GIT_REVISION")] = sha(torch / "GIT_REVISION")
    frozen = Path("/opt/bc250-stack/previous")
    frozen_manifest = json.loads(
        (frozen / "review-20261013/runtime-sha256.json").read_text()
    )
    for n in ["rocm/lib/libhipblaslt.so.1.5", "rocm/.kpack/blas_lib_gfx1013.kpack"]:
        protected[str(frozen / n)] = frozen_manifest[n]
    for p, h in protected.items():
        assert sha(Path(p)) == h, p
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    states = []
    for label in labels + ["dense-int8-a08-candidate"]:
        state = json.loads((Path("/run") / label / "state.json").read_text())
        assert (
            state["status"] == "acknowledged"
            and state["boot_id"] == boot_id
            and state["timeout_seconds"] == 600
        )
        states.append(dict(label=label, status=state["status"], timeout_seconds=600))
    recovery = WS / "recovery/dense-int8-a08-before"
    metadata = RELEASE / "runtime.json"
    old_metadata = metadata.read_bytes()
    stamp = WS / "source/rocm-libraries/GIT_REVISION"
    old_stamp = stamp.read_bytes()
    assert old_stamp.decode().strip() == source["parent_libraries_revision"]
    assert not (recovery / "runtime.json").exists()
    (recovery / "runtime.json").write_bytes(old_metadata)
    (recovery / "GIT_REVISION").write_bytes(old_stamp)
    report = dict(
        status="passed",
        source=source,
        artifacts=after,
        protected_sha256=protected,
        protected_unchanged=True,
        gates=labels,
        guard_states=states,
        kernel=os.uname().release,
        boot_id=boot_id,
        current_torch=str(torch),
        running_agent_jobs=[],
        disk_available_bytes=os.statvfs(WS).f_bavail * os.statvfs(WS).f_frsize,
    )
    try:
        runtime = json.loads(old_metadata)
        runtime["previous_a08_libraries_revision"] = runtime["rocm_libraries_revision"]
        runtime["rocm_libraries_revision"] = source["libraries_revision"]
        runtime["effective_therock_port_revision"] = source["therock_port_revision"]
        runtime[
            "source_state"
        ] += "; native signed INT8/INT32 scalar Lt route and existing Torch _int_mm qualified"
        runtime["a08_dense_int8"] = dict(
            status="bounded installed integer/floating/model/profiler gates passed",
            source_manifest=str(WS / "SOURCE_DENSE_INT8_A08.json"),
            artifacts=after,
            contract="COMPUTE_32I, INT32 host/device scalars, modulo32 accumulation/output; DEFAULT/BIAS/RELU/RELU_BIAS; no quantization integration or universal speedup",
        )
        stamp.write_text(source["libraries_revision"] + "\n")
        atomic_json(metadata, runtime)
        for p in [
            WS / "SOURCE_DENSE_INT8_A08.json",
            RELEASE / "SOURCE_DENSE_INT8_A08.json",
            WS / "logs/dense-int8-a08-final.json",
        ]:
            atomic_json(p, report)
    except BaseException:
        metadata.write_bytes(old_metadata)
        stamp.write_bytes(old_stamp)
        raise
    print(
        "DENSE_INT8_A08_INSTALLED_SOURCE_ARTIFACT_AND_PROTECTED_HASHES_PASS",
        len(protected),
    )


if __name__ == "__main__":
    main()
