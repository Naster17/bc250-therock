# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise actual BLAS declarations without compiling their dependencies."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class Bc250BlasConfigTest(unittest.TestCase):
    def _configure(self, targets: str) -> dict[str, str]:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            script = r"""
cmake_minimum_required(VERSION 3.25)
project(BlasDeclarationProbe NONE)
include("@ROOT@/cmake/therock_amdgpu_targets.cmake")
include("@ROOT@/cmake/therock_subproject.cmake")
set(THEROCK_AMDGPU_TARGETS "@TARGETS@")
set(THEROCK_BUILD_TESTING ON)
set(THEROCK_ENABLE_SPARSE ON)
set(THEROCK_ROCM_LIBRARIES_SOURCE_DIR "@ROOT@/rocm-libraries")

function(therock_cmake_subproject_declare name)
  cmake_parse_arguments(ARG "BACKGROUND_BUILD;DISABLE_AMDGPU_TARGETS" ""
    "EXTERNAL_SOURCE_DIR;BINARY_DIR;DEFAULT_GPU_TARGETS;CMAKE_ARGS;CMAKE_INCLUDES;COMPILER_TOOLCHAIN;BUILD_DEPS;RUNTIME_DEPS" ${ARGN})
  set_property(GLOBAL PROPERTY "probe_${name}" TRUE)
  file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/${name}.args" CONTENT "${ARG_CMAKE_ARGS}")
  file(WRITE "${CMAKE_BINARY_DIR}/${name}.deps" "${ARG_BUILD_DEPS};${ARG_RUNTIME_DEPS}")
  file(WRITE "${CMAKE_BINARY_DIR}/${name}.targets" "${ARG_DISABLE_AMDGPU_TARGETS};${ARG_DEFAULT_GPU_TARGETS}")
endfunction()
function(therock_cmake_subproject_dist_dir output name)
  set(${output} "${CMAKE_BINARY_DIR}/hip" PARENT_SCOPE)
endfunction()
foreach(command therock_cmake_subproject_activate
    therock_cmake_subproject_glob_c_sources
    therock_cmake_subproject_provide_package therock_provide_artifact)
  function(${command})
  endfunction()
endforeach()
include("@ROOT@/math-libs/BLAS/CMakeLists.txt")
get_property(roller GLOBAL PROPERTY probe_rocRoller)
file(WRITE "${CMAKE_BINARY_DIR}/roller" "${roller}")
get_property(sparse_lt GLOBAL PROPERTY probe_hipSPARSELt)
file(WRITE "${CMAKE_BINARY_DIR}/sparse_lt" "${sparse_lt}")
"""
            (source / "CMakeLists.txt").write_text(
                script.replace("@ROOT@", ROOT.as_posix()).replace("@TARGETS@", targets)
            )
            build = source / "build"
            result = subprocess.run(
                ["cmake", "-S", str(source), "-B", str(build)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return {
                name: (build / name).read_text()
                for name in (
                    "hipBLASLt.args",
                    "hipBLASLt.deps",
                    "hipBLASLt.targets",
                    "rocBLAS.args",
                    "rocBLAS.deps",
                    "roller",
                    "sparse_lt",
                )
            }

    def test_bc250_uses_native_host_api_and_tensile(self):
        config = self._configure("gfx1013")
        self.assertIn("-DHIPBLASLT_HOST_API_ONLY=ON", config["hipBLASLt.args"])
        self.assertIn("-DGPU_TARGETS=gfx1013", config["hipBLASLt.args"])
        self.assertIn("-DHIPBLASLT_ENABLE_DEVICE=OFF", config["hipBLASLt.args"])
        self.assertEqual(config["hipBLASLt.targets"], "TRUE;")
        self.assertNotIn("mxDataGenerator", config["hipBLASLt.deps"])
        self.assertNotIn("amdsmi", config["hipBLASLt.deps"])
        self.assertEqual(config["roller"], "")
        self.assertIn("-DBUILD_WITH_TENSILE=ON", config["rocBLAS.args"])
        self.assertIn("-DBUILD_WITH_HIPBLASLT=0", config["rocBLAS.args"])
        self.assertNotIn("hipBLASLt", config["rocBLAS.deps"])
        self.assertEqual(config["sparse_lt"], "")

    def test_supported_sparse_lt_is_still_declared(self):
        config = self._configure("gfx942")
        self.assertEqual(config["sparse_lt"], "TRUE")

    def test_sparse_artifacts_allow_excluded_lt_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "build_tools/fileset_tool.py"),
                    "artifact",
                    "--root-dir",
                    str(root),
                    "--descriptor",
                    str(ROOT / "math-libs/BLAS/artifact-sparse.toml"),
                    "--artifact-name",
                    "sparse",
                    "lib",
                    str(root / "lib"),
                    "test",
                    str(root / "test"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_supported_and_mixed_targets_keep_device_backend(self):
        for targets in ("gfx1100", "gfx1013;gfx1100"):
            with self.subTest(targets=targets):
                config = self._configure(targets)
                self.assertNotIn(
                    "-DHIPBLASLT_HOST_API_ONLY=ON", config["hipBLASLt.args"]
                )
                self.assertIn(
                    "-DHIPBLASLT_ENABLE_ROCROLLER=ON", config["hipBLASLt.args"]
                )
                self.assertEqual(config["hipBLASLt.targets"], "FALSE;gfx1100")
                self.assertEqual(config["roller"], "TRUE")
                self.assertIn("-DBUILD_WITH_HIPBLASLT=1", config["rocBLAS.args"])
                self.assertIn("hipBLASLt", config["rocBLAS.deps"])


if __name__ == "__main__":
    unittest.main()
