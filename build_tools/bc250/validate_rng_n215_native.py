#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Direct rocRAND offsets, reset/stream contracts and integer CPU Philox oracle."""

import argparse
import ctypes
import itertools
import json
import math
from pathlib import Path


def philox_words(seed: int, offset: int, count: int) -> list[int]:
    """Scalar Random123 Philox4x32-10; 64-bit seed, zero subsequence."""
    mask = 2**32 - 1
    words = []
    for block in range(offset // 4, (offset + count + 3) // 4):
        counter = [block & mask, (block >> 32) & mask, 0, 0]
        key = [seed & mask, seed >> 32]
        for _ in range(10):
            p0, p1 = 0xD2511F53 * counter[0], 0xCD9E8D57 * counter[2]
            counter = [
                (p1 >> 32) ^ counter[1] ^ key[0],
                p1 & mask,
                (p0 >> 32) ^ counter[3] ^ key[1],
                p0 & mask,
            ]
            key = [(key[0] + 0x9E3779B9) & mask, (key[1] + 0xBB67AE85) & mask]
        words.extend(counter)
    return words[offset % 4 : offset % 4 + count]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(4)
    arch = torch.cuda.get_device_properties(0).gcnArchName
    assert arch.split(":")[0] == "gfx1013", arch
    lib = ctypes.CDLL("librocrand.so")
    P, I, U, L, S, D = (
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.c_ulonglong,
        ctypes.c_size_t,
        ctypes.c_double,
    )
    signatures = {
        "create_generator": [ctypes.POINTER(P), I],
        "destroy_generator": [P],
        "set_seed": [P, L],
        "set_offset": [P, L],
        "set_ordering": [P, I],
        "set_stream": [P, P],
        "set_quasi_random_generator_dimensions": [P, U],
        "generate": [P, P, S],
        "generate_uniform_double": [P, P, S],
        "generate_normal_double": [P, P, S, D, D],
        "generate_log_normal_double": [P, P, S, D, D],
        "generate_poisson": [P, P, S, D],
    }
    for name, signature in signatures.items():
        fn = getattr(lib, "rocrand_" + name)
        fn.argtypes, fn.restype = signature, I

    def call(name, *values):
        status = getattr(lib, "rocrand_" + name)(*values)
        assert status == 0, (name, status)

    seed = 2**63 + 10132150
    report = dict(status="running", arch=arch, checks=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def check(name, execute):
        try:
            row = dict(name=name, status="passed", details=execute())
        except Exception as error:
            row = dict(name=name, status="failed", error=repr(error))
        report["checks"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(row, flush=True)

    def generate(g, kind, n=65536):
        integer = kind in ("raw", "poisson")
        dtype = torch.int32 if integer else torch.float64
        # 8-element prefix aligns normal's double2 writes; protect exact exterior.
        guard = torch.full((n + 16,), 123, device="cuda", dtype=dtype)
        out = guard[8:-8]
        fn, extra = {
            "raw": ("generate", []),
            "uniform": ("generate_uniform_double", []),
            "normal": ("generate_normal_double", [0.2, 0.7]),
            "lognormal": ("generate_log_normal_double", [0.2, 0.4]),
            "poisson": ("generate_poisson", [4.5]),
        }[kind]
        call(fn, g, out.data_ptr(), n, *extra)
        torch.cuda.current_stream().synchronize()
        host = guard.cpu()
        assert torch.all(host[:8] == 123) and torch.all(host[-8:] == 123)
        result = host[8:-8]
        return (result.to(torch.int64) & 0xFFFFFFFF) if integer else result

    for rng, kind, offset in itertools.product(
        (401, 402, 404, 405, 406),
        ("uniform", "normal", "lognormal", "poisson"),
        (17, 2**32 + 17),
    ):

        def reset_stream():
            g = P()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            try:
                call("create_generator", ctypes.byref(g), rng)
                call("set_seed", g, seed)
                outputs = []
                for private in (False, True):
                    call("set_offset", g, offset)
                    active = stream if private else torch.cuda.current_stream()
                    call("set_stream", g, active.cuda_stream)
                    with torch.cuda.stream(active):
                        outputs.append(generate(g, kind))
                torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
                values = outputs[0].double()
                assert torch.isfinite(values).all()
                mean, var = {
                    "uniform": (0.5, 1 / 12),
                    "normal": (0.2, 0.7**2),
                    "lognormal": (
                        math.exp(0.28),
                        (math.exp(0.16) - 1) * math.exp(0.56),
                    ),
                    "poisson": (4.5, 4.5),
                }[kind]
                if kind == "uniform":
                    assert values.min() > 0 and values.max() <= 1
                elif kind in ("poisson", "lognormal"):
                    assert values.min() >= 0
                observed = [values.mean().item(), values.var(unbiased=False).item()]
                assert abs(observed[0] - mean) < 8 * math.sqrt(var / len(values))
                assert abs(observed[1] - var) < 0.06 * max(var, 0.1)
                return dict(
                    offset=offset,
                    exact_reset_replay=True,
                    stream_rebound=True,
                    observed=observed,
                    expected=[mean, var],
                    sentinels_unchanged=True,
                )
            finally:
                torch.cuda.synchronize()
                if g.value:
                    call("destroy_generator", g)

        check(f"offset_reset_stream_{rng}_{kind}_{offset}", reset_stream)

    # Legacy Philox is a linear sequence, unlike architecture-tuned default
    # orderings or Torch's per-thread offset. Compare exact integers to CPU.
    for offset in (0, 1, 3, 4, 17, 2**32 + 3, 2**40 + 1):

        def philox():
            g = P()
            try:
                call("create_generator", ctypes.byref(g), 404)
                call("set_ordering", g, 103)
                call("set_seed", g, seed)
                call("set_offset", g, offset)
                actual = generate(g, "raw", 257)
                expected = torch.tensor(
                    philox_words(seed, offset, 257), dtype=torch.int64
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                return dict(offset=offset, exact_cpu_words=257)
            finally:
                torch.cuda.synchronize()
                if g.value:
                    call("destroy_generator", g)

        check(f"legacy_philox_cpu_{offset}", philox)

    for rng, dims, offset in itertools.product((501, 502, 504, 505), (1, 3), (0, 17)):

        def quasi():
            g = P()
            try:
                call("create_generator", ctypes.byref(g), rng)
                call("set_quasi_random_generator_dimensions", g, dims)
                call("set_offset", g, 0)
                prefix = generate(g, "uniform", dims * (offset + 257)).reshape(
                    dims, -1
                )[:, offset:]
                call("set_offset", g, offset)
                actual = generate(g, "uniform", dims * 257).reshape(dims, -1)
                torch.testing.assert_close(actual, prefix, rtol=0, atol=0)
                call("set_offset", g, offset)
                replay = generate(g, "uniform", dims * 257).reshape(dims, -1)
                torch.testing.assert_close(actual, replay, rtol=0, atol=0)
                return dict(
                    dimensions=dims,
                    offset=offset,
                    exact_prefix_slice=True,
                    exact_reset_replay=True,
                )
            finally:
                torch.cuda.synchronize()
                if g.value:
                    call("destroy_generator", g)

        check(f"sobol_per_dimension_offset_{rng}_{dims}_{offset}", quasi)

    def lfsr_prefix():
        g = P()
        try:
            call("create_generator", ctypes.byref(g), 406)
            call("set_ordering", g, 103)
            call("set_seed", g, seed)
            prefix = generate(g, "raw", 274)[17:]
            call("set_offset", g, 17)
            actual = generate(g, "raw", 257)
            torch.testing.assert_close(actual, prefix, rtol=0, atol=0)
            call("set_offset", g, 2**32 + 17)
            high = generate(g, "raw", 257)
            assert not torch.equal(high, actual), "High offset truncated to low 32 bits"
            return dict(exact_prefix_slice=True, high_offset_does_not_alias_low=True)
        finally:
            torch.cuda.synchronize()
            if g.value:
                call("destroy_generator", g)

    check("lfsr113_legacy_prefix_and_high_offset", lfsr_prefix)

    for rng in (403,):

        def unsupported():
            g = P()
            try:
                call("create_generator", ctypes.byref(g), rng)
                status = lib.rocrand_set_offset(g, 17)
                assert status == 103, status
                call("set_seed", g, 10132150)
                assert generate(g, "uniform", 1024).isfinite().all()
                return dict(
                    offset_status=status,
                    documented_type_error=True,
                    usable_after_refusal=True,
                )
            finally:
                torch.cuda.synchronize()
                if g.value:
                    call("destroy_generator", g)

        check(f"unsupported_offset_{rng}", unsupported)

    wrapper = ctypes.CDLL("libhiprand.so")
    for name, signature in {
        "CreateGenerator": [ctypes.POINTER(P), I],
        "DestroyGenerator": [P],
        "SetPseudoRandomGeneratorSeed": [P, L],
        "SetGeneratorOffset": [P, L],
        "SetGeneratorOrdering": [P, I],
        "Generate": [P, P, S],
    }.items():
        fn = getattr(wrapper, "hiprand" + name)
        fn.argtypes, fn.restype = signature, I

    def hipcall(name, *values):
        status = getattr(wrapper, "hiprand" + name)(*values)
        assert status == 0, (name, status)

    for offset in (0, 17, 2**40 + 1):

        def hiprand_oracle():
            g = P()
            try:
                hipcall("CreateGenerator", ctypes.byref(g), 405)
                hipcall("SetGeneratorOrdering", g, 103)
                hipcall("SetPseudoRandomGeneratorSeed", g, seed)
                hipcall("SetGeneratorOffset", g, offset)
                guard = torch.full((273,), 123, dtype=torch.int32, device="cuda")
                hipcall("Generate", g, guard[8:-8].data_ptr(), 257)
                torch.cuda.synchronize()
                host = guard.cpu()
                assert torch.all(host[:8] == 123) and torch.all(host[-8:] == 123)
                actual = host[8:-8].to(torch.int64) & 0xFFFFFFFF
                expected = torch.tensor(
                    philox_words(seed, offset, 257), dtype=torch.int64
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                return dict(
                    offset=offset, exact_cpu_words=257, correct_wrapper_enum=True
                )
            finally:
                torch.cuda.synchronize()
                if g.value:
                    hipcall("DestroyGenerator", g)

        check(f"hiprand_legacy_philox_cpu_{offset}", hiprand_oracle)

    report["status"] = (
        "passed" if all(r["status"] == "passed" for r in report["checks"]) else "failed"
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    assert report["status"] == "passed"


if __name__ == "__main__":
    main()
