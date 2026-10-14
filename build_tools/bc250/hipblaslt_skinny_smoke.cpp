// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <string>
#define main bc250LtScalarMain
#include "hipblaslt_scalar_smoke.cpp"
#undef main

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  hipDeviceProp_t props{};
  hip_check(hipGetDeviceProperties(&props, 0));
  if (std::string(props.gcnArchName).substr(0, 7) != "gfx1013")
    return 1;
  const bool timing = argc > 1 && std::string(argv[1]) == "--timing";
  hipblasLtHandle_t handle{};
  lt_check(hipblasLtCreate(&handle));
  int passed = 0;
  auto shapes =
      timing ? std::vector<std::vector<int>>{{1, 257, 64}, {257, 1, 64},
                                             {4, 257, 64}, {257, 4, 64},
                                             {17, 19, 23}, {32, 32, 32},
                                             {5, 257, 64}, {4, 257, 65},
                                             {32, 33, 32}, {128, 128, 64},
                                             {16, 32, 64}, {17, 32, 64}}
             : std::vector<std::vector<int>>{
                   {1, 257, 0},  {257, 1, 1},  {3, 85, 15},  {4, 65, 16},
                   {65, 4, 17},  {4, 257, 63}, {257, 4, 64}, {4, 257, 65},
                   {5, 65, 64},  {31, 32, 63}, {32, 32, 64}, {32, 33, 64},
                   {16, 32, 64}, {17, 32, 64}, {32, 16, 64}, {32, 17, 64}};
  for (auto shape : shapes)
    for (bool row : {false, true})
      for (bool ta : {false, true})
        for (bool tb : {false, true}) {
          if (timing && (row || ta || tb))
            continue;
          const int m = shape[0], n = shape[1], k = shape[2];
          auto execute = [&](auto input, auto output, hipDataType it,
                             hipDataType ot) {
            using I = decltype(input);
            using O = decltype(output);
            run<I, O>(handle, it, ot, row, ta, tb, timing ? 1 : 3, ta, tb,
                      !timing && ta && tb, row, passed, k, !timing && k == 1,
                      HIPBLASLT_EPILOGUE_DEFAULT, HIPBLASLT_DATATYPE_INVALID, m,
                      n, timing);
            std::printf("PASS skinny m=%d n=%d k=%d input=%d output=%d\n", m, n,
                        k, int(it), int(ot));
          };
          execute(float{}, float{}, HIP_R_32F, HIP_R_32F);
          execute(__half{}, __half{}, HIP_R_16F, HIP_R_16F);
          execute(hip_bfloat16{}, hip_bfloat16{}, HIP_R_16BF, HIP_R_16BF);
          if (!timing) {
            execute(__half{}, float{}, HIP_R_16F, HIP_R_32F);
            execute(hip_bfloat16{}, float{}, HIP_R_16BF, HIP_R_32F);
          }
        }
  // Bias + activation at skinny dimensions, mixed storage/orders/transposes,
  // device scalars and broadcast inputs. Full extended epilogues run
  // separately.
  if (!timing)
    for (auto epi : {HIPBLASLT_EPILOGUE_BIAS, HIPBLASLT_EPILOGUE_RELU_BIAS,
                     HIPBLASLT_EPILOGUE_GELU_BIAS})
      for (bool row : {false, true})
        for (bool ta : {false, true})
          for (bool tb : {false, true}) {
            run<float, float>(handle, HIP_R_32F, HIP_R_32F, row, ta, tb, 3,
                              false, true, false, false, passed, 64, false, epi,
                              HIP_R_32F, 4, 65);
            std::puts("PASS skinny epilogue");
          }
  lt_check(hipblasLtDestroy(handle));
  std::printf("HIPBLASLT_SKINNY_PASS cases=%d\n", passed);
}
