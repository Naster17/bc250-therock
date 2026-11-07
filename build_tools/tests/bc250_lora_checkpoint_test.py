"""Checkpoint transport refusal and original-file preservation tests; no pickle/GPU."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

MODULE = Path(__file__).resolve().parents[1] / "bc250/lora_resume/checkpoint_files.py"
spec = importlib.util.spec_from_file_location("lora_checkpoint_files", MODULE)
files = importlib.util.module_from_spec(spec)
spec.loader.exec_module(files)
guard_spec = importlib.util.spec_from_file_location(
    "lora_guard", MODULE.with_name("guard.py")
)
guard = importlib.util.module_from_spec(guard_spec)
guard_spec.loader.exec_module(guard)


class CheckpointTransport(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.master = self.root / "master"
        self.peer = self.root / "peer"
        self.master.mkdir()
        self.peer.mkdir()
        for name in files.REQUIRED:
            (self.master / name).write_bytes(b"owned payload: " + name.encode())
        (self.master / "trainer_state.json").write_text('{"global_step": 2}')
        (self.master / "rng_state_0.pth").write_bytes(b"rank0")
        (self.peer / "rng_state_1.pth").write_bytes(b"rank1")
        state = dict(step=2, parameters="p", optimizer="o", scheduler="s", scaler="a")
        self.manifest = dict(
            format="bc250-lora-ddp-v1",
            world_size=2,
            step=2,
            checkpoint_id="owned-run-2",
            payload=files.common_inventory(self.master),
            ranks={
                str(rank): dict(state=state, rng_file=files.file_info(path))
                for rank, path in [
                    (0, self.master / "rng_state_0.pth"),
                    (1, self.peer / "rng_state_1.pth"),
                ]
            },
        )
        for path in [self.master, self.peer]:
            (path / files.MANIFEST).write_text(json.dumps(self.manifest))
        self.archive = self.root / "payload.tar"
        files.export(self.master, self.archive)

    def test_complete_copy_preserves_both_originals_and_rng(self):
        before = {p.name: p.read_bytes() for p in self.peer.iterdir()}
        target = self.root / "ready"
        files.prepare(self.archive, self.peer, target, 1)
        self.assertEqual(files.validate(target, 1), self.manifest)
        self.assertFalse((target / "rng_state_0.pth").exists())
        self.assertEqual({p.name: p.read_bytes() for p in self.peer.iterdir()}, before)
        self.assertEqual(files.validate(self.master, 0), self.manifest)

    def test_missing_optimizer_refused(self):
        (self.master / "optimizer.pt").unlink()
        with self.assertRaises(AssertionError):
            files.validate(self.master, 0)

    def test_corrupt_scaler_refused(self):
        (self.master / "scaler.pt").write_bytes(b"modified")
        with self.assertRaises(AssertionError):
            files.validate(self.master, 0)

    def test_wrong_rank_rng_refused(self):
        (self.peer / "rng_state_1.pth").write_bytes(b"rank0")
        with self.assertRaises(AssertionError):
            files.prepare(self.archive, self.peer, self.root / "ready", 1)
        self.assertFalse((self.root / "ready").exists())

    def test_mixed_generation_never_published(self):
        changed = dict(self.manifest, checkpoint_id="another-run-2")
        (self.peer / files.MANIFEST).write_text(json.dumps(changed))
        with self.assertRaises(AssertionError):
            files.prepare(self.archive, self.peer, self.root / "ready", 1)
        self.assertFalse((self.root / "ready").exists())
        self.assertEqual((self.peer / "rng_state_1.pth").read_bytes(), b"rank1")

    def test_existing_destination_preserved(self):
        target = self.root / "ready"
        target.mkdir()
        (target / "user-file").write_text("keep")
        with self.assertRaises(AssertionError):
            files.prepare(self.archive, self.peer, target, 1)
        self.assertEqual((target / "user-file").read_text(), "keep")

    def test_symlink_payload_refused(self):
        (self.master / "optimizer.pt").unlink()
        (self.master / "optimizer.pt").symlink_to(self.master / "scheduler.pt")
        with self.assertRaises(AssertionError):
            files.validate(self.master, 0)

    def test_other_world_refused(self):
        changed = dict(self.manifest, world_size=1)
        (self.peer / files.MANIFEST).write_text(json.dumps(changed))
        with self.assertRaises(AssertionError):
            files.prepare(self.archive, self.peer, self.root / "ready", 1)


class DockerPreflight(unittest.TestCase):
    def test_absent_socket_does_not_query_remote_daemon(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(
            guard.subprocess, "run"
        ) as run:
            self.assertEqual(guard.docker_containers(Path(temp) / "absent"), [])
            run.assert_not_called()

    def test_local_containers_enumerated(self):
        with tempfile.TemporaryDirectory() as temp:
            socket = Path(temp) / "docker.sock"
            socket.touch()
            result = guard.subprocess.CompletedProcess(
                [], 0, stdout="b\na\n", stderr=""
            )
            with mock.patch.object(
                guard.shutil, "which", return_value="/usr/bin/docker"
            ), mock.patch.object(guard.subprocess, "run", return_value=result) as run:
                self.assertEqual(guard.docker_containers(socket), ["a", "b"])
                self.assertEqual(
                    run.call_args.args[0],
                    ["docker", "--host", "unix://" + str(socket), "ps", "-q"],
                )

    def test_unavailable_daemon_state_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            socket = Path(temp) / "docker.sock"
            socket.touch()
            result = guard.subprocess.CompletedProcess(
                [], 1, stdout="", stderr="unavailable"
            )
            with mock.patch.object(
                guard.shutil, "which", return_value="/usr/bin/docker"
            ), mock.patch.object(guard.subprocess, "run", return_value=result):
                with self.assertRaises(AssertionError):
                    guard.docker_containers(socket)

    def test_missing_cli_with_present_socket_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            socket = Path(temp) / "docker.sock"
            socket.touch()
            with mock.patch.object(guard.shutil, "which", return_value=None):
                with self.assertRaises(AssertionError):
                    guard.docker_containers(socket)


if __name__ == "__main__":
    unittest.main()
