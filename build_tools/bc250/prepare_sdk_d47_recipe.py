#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Parse retained compile/link recipes as argv; check inputs without compiling."""
import json
from pathlib import Path
import shlex

WORK = Path("/home/nik/bc250work")
ROOT = WORK / "evidence/source-d47-r1"


def main():
    budget = json.loads((ROOT / "sdk-build-admission.json").read_text())
    commands = budget["agent_compile_commands"]
    assert len(commands) == 1
    compile_args = shlex.split(commands[0]["command"])
    lines = (ROOT / "sdk-target-commands.log").read_text().splitlines()
    links = [line for line in lines if "-o lib/librocprofiler-sdk.so.1.4.2" in line]
    assert len(links) == 1
    shell_tokens = shlex.split(links[0])
    start = next(
        i for i, value in enumerate(shell_tokens) if value.endswith("/clang++")
    )
    end = (
        shell_tokens.index("&&", start)
        if "&&" in shell_tokens[start:]
        else len(shell_tokens)
    )
    link_args = shell_tokens[start:end]
    assert not any(value in ("&&", ";", "|", ">") for value in compile_args + link_args)
    target = "source/lib/rocprofiler-sdk/CMakeFiles/rocprofiler-sdk-object-library.dir/agent.cpp.o"
    assert link_args.count(target) == 1
    assert compile_args.count(target) == 1
    build = Path(budget["build"])

    def host_path(name):
        if name.startswith("/workspace/"):
            return WORK / "rocm-10.2-gfx1013" / name.removeprefix("/workspace/")
        path = Path(name)
        return path if path.is_absolute() else build / path

    inputs = []
    output_index = link_args.index("-o") + 1
    for index, value in enumerate(link_args):
        if (
            index != output_index
            and not value.startswith("-")
            and (value.endswith((".o", ".a")) or ".so" in Path(value).name)
        ):
            path = host_path(value)
            inputs.append(
                dict(argument=value, host_path=str(path), present=path.is_file())
            )
    compiler = next(arg for arg in compile_args if arg.endswith("/clang++"))
    recipe = dict(
        status=(
            "argv-parsed-inputs-present"
            if all(x["present"] for x in inputs) and host_path(compiler).is_file()
            else "missing-inputs"
        ),
        compile_argv=compile_args,
        link_argv=link_args,
        retained_working_directory=str(build),
        agent_object=target,
        inputs=inputs,
        compiler=str(host_path(compiler)),
        compiler_present=host_path(compiler).is_file(),
        restrictions=[
            "Do not execute these original argv: they target the retained build output.",
            "Candidate must redirect source/object/depfile/library outputs into an owned directory.",
            "Read-only original build/source mounts; matching recorded image; ccache20G/jobs1/link1.",
            "Resource admission is provisional; recheck locks, ownership and actual resources before build/tests.",
        ],
        native_build_started=False,
    )
    (ROOT / "sdk-recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                status=recipe["status"],
                inputs=len(inputs),
                missing=[x["host_path"] for x in inputs if not x["present"]],
            )
        )
    )


if __name__ == "__main__":
    main()
