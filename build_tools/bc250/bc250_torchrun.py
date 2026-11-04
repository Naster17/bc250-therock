#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run torchrun with the qualified source-derived agent exit-lifetime repair."""
import ast
import hashlib
import json
from pathlib import Path
import runpy

import torch.distributed.elastic.agent.server.api as agent_api


def install_exit_fix():
    base = Path(__file__).resolve().parent
    manifest = json.loads((base / "torchrun-fix.json").read_text())
    source = base / "agent-api-source.py"
    actual = hashlib.sha256(Path(agent_api.__file__).read_bytes()).hexdigest()
    if actual != manifest["original_agent_api_sha256"]:
        raise RuntimeError(
            "Unqualified Torch agent API; requalify the distributed launcher"
        )
    if (
        hashlib.sha256(source.read_bytes()).hexdigest()
        != manifest["patched_agent_api_sha256"]
    ):
        raise RuntimeError("Distributed agent repair source identity mismatch")
    tree = ast.parse(source.read_text())
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
    scope = dict(vars(agent_api))
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
            str(source),
            "exec",
        ),
        scope,
    )
    agent_api.SimpleElasticAgent._exit_barrier = scope["_exit_barrier"]


if __name__ == "__main__":
    install_exit_fix()
    runpy.run_module("torch.distributed.run", run_name="__main__")
