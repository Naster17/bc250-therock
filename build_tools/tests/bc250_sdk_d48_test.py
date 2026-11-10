# SPDX-License-Identifier: MIT
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "bc250"
spec = importlib.util.spec_from_file_location(
    "audit_elf_d46", SCRIPTS / "audit_elf_d46.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
sys.modules.setdefault("audit_elf_d46", audit)
spec = importlib.util.spec_from_file_location(
    "build_sdk_d48", SCRIPTS / "build_sdk_d48_in_container.py"
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class StopAfterLink(Exception):
    pass


def recipe():
    target = "source/lib/rocprofiler-sdk/CMakeFiles/rocprofiler-sdk-object-library.dir/agent.cpp.o"
    return dict(
        agent_object=target,
        compile_argv=[
            "clang++",
            "-DROCPROFILER_CI=1",
            "-o",
            target,
            "-c",
            "/workspace/source/rocm-systems/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/agent.cpp",
        ],
        link_argv=[
            "clang++",
            target,
            "-o",
            "lib/librocprofiler-sdk.so.1.4.2",
            "-Xlinker",
            "--dependency-file=source/retained/link.d",
        ],
    )


class SDKBuildTest(unittest.TestCase):
    def prepare(self, root, value):
        (root / "sdk-recipe.json").write_text(json.dumps(value))
        (root / "agent.cpp").write_text("// candidate")

    def test_object_source_library_and_link_depfile_all_redirected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root, recipe())
            calls = []

            def execute(command, **kwargs):
                calls.append(command)
                if command[0] == "clang++":
                    raise StopAfterLink()

            with patch.object(builder, "ROOT", root), patch.object(
                builder.subprocess, "run", side_effect=execute
            ):
                with self.assertRaises(StopAfterLink):
                    builder.main()
            compile_args, link_args = calls[1:3]
            self.assertIn(str(root / "agent.cpp"), compile_args)
            self.assertIn(str(root / "candidate-agent.o"), compile_args)
            self.assertIn(str(root / "candidate-agent.o"), link_args)
            self.assertIn(str(root / "librocprofiler-sdk.so.1.4.2"), link_args)
            self.assertIn(
                "--dependency-file=" + str(root / "candidate-link.d"), link_args
            )
            self.assertNotIn("--dependency-file=source/retained/link.d", link_args)

    def test_ci_diagnostics_cannot_be_silently_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = recipe()
            value["compile_argv"].remove("-DROCPROFILER_CI=1")
            self.prepare(root, value)
            with patch.object(builder, "ROOT", root), patch.object(
                builder.subprocess, "run"
            ) as run:
                with self.assertRaises(AssertionError):
                    builder.main()
                run.assert_not_called()

    def test_existing_object_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root, recipe())
            (root / "candidate-agent.o").write_bytes(b"preserve")
            with patch.object(builder, "ROOT", root), patch.object(
                builder.subprocess, "run"
            ) as run:
                with self.assertRaises(AssertionError):
                    builder.main()
                run.assert_not_called()
            self.assertEqual((root / "candidate-agent.o").read_bytes(), b"preserve")


if __name__ == "__main__":
    unittest.main()
