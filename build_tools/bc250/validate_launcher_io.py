# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check persistent training files and multiprocessing inside bc250-torch."""

import argparse
import errno
import hashlib
import json
from multiprocessing import shared_memory
from pathlib import Path
import os
import pwd
import urllib.request

import torch
from torch.utils.data import DataLoader, TensorDataset

from launcher_sidecar import SIDECAR_VALUE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("write", "resume"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--url", required=True)
    args = parser.parse_args()
    root = args.root
    assert Path.cwd() == root / "training project"
    assert Path.home() == Path(os.environ["EXPECTED_HOME"])
    account = pwd.getpwuid(os.getuid())
    assert account.pw_name == os.environ["USER"]
    assert account.pw_dir == str(Path.home())
    assert os.environ["HF_TOKEN"] == "BC250_LAUNCHER_FAKE_TOKEN"
    assert os.environ["CUSTOM_TRAINING_SETTING"] == "value with spaces=42"
    assert SIDECAR_VALUE == 1704
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1013"
    assert args.data.read_text() == "host dataset\n"
    assert Path("linked dataset").read_text() == "host dataset\n"
    try:
        args.data.write_text("must fail")
    except OSError as error:
        assert error.errno in (errno.EROFS, errno.EACCES, errno.EPERM)
    else:
        raise AssertionError("Read-only dataset mount accepted a write")
    home_cache = Path.home() / ".cache/bc250-torch/launcher-io-test"
    home_cache.mkdir(parents=True, exist_ok=True)
    output = Path("checkpoints with spaces") / "model checkpoint.pt"
    torch.manual_seed(1704)
    model = torch.nn.Linear(4, 2).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    values = torch.linspace(-1, 1, 64).reshape(16, 4).cuda()
    targets = torch.stack((values.sum(1) * 0.3, values.sum(1) * -0.2), dim=1)
    if args.phase == "write":
        output.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(12):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(values) - targets).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "output": model(values).detach().cpu(),
                "steps": 12,
                "rng": torch.get_rng_state(),
                "gpu_rng": torch.cuda.get_rng_state(),
            },
            output.with_suffix(".tmp"),
        )
        output.with_suffix(".tmp").replace(output)
        torch.save(torch.arange(19), root / "absolute tensor.pt")
        payload = urllib.request.urlopen(args.url, timeout=10).read()
        (home_cache / "download.bin").write_bytes(payload)
        (root / "download.sha256").write_text(hashlib.sha256(payload).hexdigest())
        # Touch >64 MiB: this would exhaust Podman's default shared memory.
        segment = shared_memory.SharedMemory(create=True, size=80 * 1024 * 1024)
        try:
            segment.buf[:] = b"x" * len(segment.buf)
            assert segment.buf[-1] == ord("x")
        finally:
            segment.close()
            segment.unlink()
        features = torch.arange(128 * 32).reshape(128, 32)
        loader = DataLoader(
            TensorDataset(features),
            batch_size=16,
            num_workers=2,
            multiprocessing_context="spawn",
            persistent_workers=True,
            pin_memory=True,
        )
        for _ in range(2):
            actual = torch.cat([batch[0] for batch in loader])
            torch.testing.assert_close(actual, features)
        del loader
    else:
        assert (
            root / "host-added.txt"
        ).read_text() == "visible after container restart\n"
        checkpoint = torch.load(
            output, map_location="cpu", weights_only=True, mmap=True
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["rng"])
        torch.cuda.set_rng_state(checkpoint["gpu_rng"])
        torch.testing.assert_close(model(values).cpu(), checkpoint["output"])
        optimizer.zero_grad(set_to_none=True)
        loss = (model(values) - targets).square().mean()
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss)
        assert all(state["step"].item() == 13 for state in optimizer.state.values())
        torch.testing.assert_close(
            torch.load(root / "absolute tensor.pt", weights_only=True), torch.arange(19)
        )
        assert (
            hashlib.sha256((home_cache / "download.bin").read_bytes()).hexdigest()
            == (root / "download.sha256").read_text()
        )
        (root / "resume passed.json").write_text(
            json.dumps({"steps": 13, "loss": loss.item()})
        )
    print(f"LAUNCHER_IO_{args.phase.upper()}_PASS", flush=True)


if __name__ == "__main__":
    main()
