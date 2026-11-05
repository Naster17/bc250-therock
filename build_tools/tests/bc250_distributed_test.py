# SPDX-License-Identifier: MIT
"""D09 parser and root handoff contract regressions, without Podman or GPU."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bc250"))
import bc250_distributed as cli
from bc250_torch import read_context


class DistributedOptionsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.tool = self.root / "tools"
        self.tool.mkdir()
        self.script = self.root / "script dir" / "train.py"
        self.script.parent.mkdir()
        self.script.write_text("pass\n")
        self.basic = ["--node-rank", "0"]

    def tearDown(self):
        self.tmp.cleanup()

    def parse(self, options, arguments=None, environment=None):
        return cli.parse_distributed_arguments(
            self.basic + options + [str(self.script)] + (arguments or []),
            environment or {},
            self.tool,
        )

    def test_order_and_spaced_argv(self):
        env = self.root / "vars.env"
        env.write_text("# owned\nX=from-file\nPYTHONPATH=/some consumer path\n")
        p = self.parse(
            [
                "--env",
                "X=before",
                "--env-file",
                str(env),
                "--env",
                "X=after",
                "--env",
                "FROM_CALLER",
            ],
            ["two words", "--env", "SCRIPT=value"],
            {"FROM_CALLER": "caller value"},
        )
        self.assertEqual(p.environment["X"], "after")
        self.assertEqual(p.environment["FROM_CALLER"], "caller value")
        self.assertEqual(p.arguments[-3:], ["two words", "--env", "SCRIPT=value"])
        self.assertEqual(p.environment["PYTHONPATH"], "/some consumer path")
        self.assertEqual(p.runtime_profile, "rccl-single")
        self.assertEqual(p.environment["GLOO_SOCKET_IFNAME"], "eth0")

    def test_readonly_ancestor_preserved(self):
        with self.assertRaises(ValueError):
            self.parse(["--mount-ro", str(self.root)])

    def test_readonly_script_parent(self):
        p = self.parse(["--mount-ro", str(self.script.parent)])
        self.assertEqual(dict(p.mounts)[str(self.script.parent)], "ro")

    def test_protected_environment_and_network(self):
        for key, value in [
            ("LD_LIBRARY_PATH", "/tmp"),
            ("LD_PRELOAD", "/tmp/x"),
            ("HSA_OVERRIDE_GFX_VERSION", "10.3.0"),
            ("NCCL_P2P_DISABLE", "0"),
            ("NCCL_ALGO", "Tree"),
            ("TORCH_NCCL_ASYNC_ERROR_HANDLING", "0"),
            ("GLOO_SOCKET_IFNAME", "lo"),
            ("MASTER_ADDR", "elsewhere"),
            ("WORLD_SIZE", "1"),
            ("PYTHON_EXEC", "/tmp/python"),
        ]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.parse(["--env", f"{key}={value}"])

    def test_mount_and_value_refusals(self):
        for options in [
            ["--mount", "/usr"],
            ["--mount", str(self.tool)],
            ["--mount", str(self.root)],
            ["--mount", str(self.root / "absent")],
            ["--env", "UNSET"],
            ["--shm-size", "0"],
            ["--env", "BAD=one\ntwo"],
        ]:
            with self.subTest(options=options), self.assertRaises(
                (ValueError, FileNotFoundError)
            ):
                self.parse(options)

    def test_private_context_still_validated(self):
        options = self.parse([])
        path = self.root / "context.json"
        path.touch(mode=0o600)
        path.write_text(
            json.dumps(dict(workspace=str(self.script.parent), options=vars(options)))
        )
        restored, cwd = read_context(path, os.getuid())
        cli.validate_distributed_options(restored, self.tool)
        self.assertEqual(cwd, self.script.parent)
        for mutate in [
            lambda p: setattr(p, "shell", True),
            lambda p: setattr(p, "runtime_profile", "daily"),
            lambda p: p.arguments.__setitem__(1, "--nnodes=1"),
            lambda p: p.arguments.__setitem__(6, str(self.script)),
            lambda p: p.environment.update(NCCL_PROTO="LL"),
            lambda p: p.mounts.clear(),
        ]:
            bad = copy.deepcopy(restored)
            mutate(bad)
            with self.assertRaises(ValueError):
                cli.validate_distributed_options(bad, self.tool)

    def test_info(self):
        self.assertIsNone(cli.parse_distributed_arguments(["--info"], {}, self.tool))

    def test_port_refusal(self):
        for port in ["1", "65536"]:
            with self.subTest(port=port), self.assertRaises(ValueError):
                self.parse(["--master-port", port])


if __name__ == "__main__":
    unittest.main()
