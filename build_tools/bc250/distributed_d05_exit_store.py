#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CPU-only real TCPStore teardown repro and candidate agent-method verification."""
import argparse
import ast
import gc
import importlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from datetime import timedelta
from types import SimpleNamespace
import unittest

import torch
import torch.distributed as dist
import torch.distributed.elastic.agent.server.api as api


def candidate(path):
    tree = ast.parse(path.read_text())
    cls = next(
        x
        for x in tree.body
        if isinstance(x, ast.ClassDef) and x.name == "SimpleElasticAgent"
    )
    method = next(
        x
        for x in cls.body
        if isinstance(x, ast.FunctionDef) and x.name == "_exit_barrier"
    )
    scope = dict(vars(api))
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
            str(path),
            "exec",
        ),
        scope,
    )
    return scope["_exit_barrier"]


class DelayStore:
    def __init__(self, store, rank):
        self.store = store
        self.rank = rank

    def __getattr__(self, name):
        return getattr(self.store, name)

    def add(self, key, value):
        if self.rank == 1:
            time.sleep(0.05)
        return self.store.add(key, value)

    def wait(self, keys, *args):
        if self.rank == 1 and any(key.endswith("/last_member") for key in keys):
            time.sleep(0.75)
        return self.store.wait(keys, *args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--unit-module", type=Path, required=True)
    p.add_argument("--child", choices=["original", "candidate"])
    p.add_argument("--rank", type=int)
    p.add_argument("--port", type=int)
    a = p.parse_args()
    torch.set_num_threads(1)
    if a.child:
        tcp = dist.TCPStore(
            "127.0.0.1",
            a.port,
            2,
            a.rank == 0,
            timedelta(seconds=8),
            wait_for_workers=True,
        )
        wrapped = DelayStore(tcp, a.rank)
        errors = []

        class Capture(logging.Handler):
            def emit(self, record):
                if record.levelno >= logging.ERROR:
                    errors.append(record.getMessage())

        api.logger.addHandler(Capture())
        agent = SimpleNamespace(
            _store=wrapped,
            _worker_group=SimpleNamespace(
                group_rank=a.rank, group_world_size=2, state="SUCCEEDED"
            ),
            _exit_barrier_timeout=4,
            _in_exit_barrier=False,
        )
        fn = (
            candidate(a.candidate)
            if a.child == "candidate"
            else api.SimpleElasticAgent._exit_barrier
        )
        fn(agent)
        a.output.write_text(
            json.dumps(
                dict(errors=errors, flag_cleared=not agent._in_exit_barrier), indent=2
            )
            + "\n"
        )
        del agent, wrapped, tcp
        gc.collect()
        return
    a.output.mkdir()
    results = []
    for mode in ["original", "candidate", "candidate"]:
        folder = a.output / (mode + "-" + str(len(results)))
        folder.mkdir()
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        children = []
        logs = []
        for rank in [0, 1]:
            log = (folder / f"rank-{rank}.log").open("w")
            logs.append(log)
            children.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--candidate",
                        str(a.candidate),
                        "--unit-module",
                        str(a.unit_module),
                        "--output",
                        str(folder / f"rank-{rank}.json"),
                        "--child",
                        mode,
                        "--rank",
                        str(rank),
                        "--port",
                        str(port),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        for child in children:
            assert child.wait(timeout=30) == 0
        for log in logs:
            log.close()
        reports = [
            json.loads((folder / f"rank-{rank}.json").read_text()) for rank in [0, 1]
        ]
        assert all(x["flag_cleared"] for x in reports)
        if mode == "original":
            assert reports[1]["errors"], "Old teardown race was not reproduced"
        else:
            assert not any(x["errors"] for x in reports)
        results.append(dict(mode=mode, reports=reports))
    api.SimpleElasticAgent._exit_barrier = candidate(a.candidate)
    spec = importlib.util.spec_from_file_location("owned_agent_test", a.unit_module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = unittest.TestSuite(
        module.SimpleElasticAgentTest(name)
        for name in [
            "test_exit_barrier_sets_and_clears_flag",
            "test_exit_barrier_clears_flag_on_timeout",
        ]
    )
    unit = unittest.TextTestRunner().run(suite)
    assert unit.wasSuccessful()
    (a.output / "final.json").write_text(
        json.dumps(
            dict(
                status="passed",
                original_reproduced=True,
                candidate_clean_repetitions=2,
                unit_checks=unit.testsRun,
                results=results,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
