// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#define main bc250SparseScalarMain
#include "hipsparselt_scalar_smoke.cpp"
#undef main

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  try {
    hipDeviceProp_t props{};
    HIP(hipGetDeviceProperties(&props, 0));
    expect(std::string(props.gcnArchName).substr(0, 7) == "gfx1013",
           "native gfx1013");
    hipsparseLtHandle_t h{};
    LT(hipsparseLtInit(&h));
    hipsparseLtMatDescriptor_t da{}, db{}, dc{}, dd{};
    auto a = layout(5, 12, HIPSPARSE_ORDER_ROW, 1);
    if (argc > 1 && std::string(argv[1]) == "--baseline") {
      expect(hipsparseLtStructuredDescriptorInit(
                 &h, &da, a.rows, a.cols, a.ld, 16, HIP_R_8I, a.order,
                 HIPSPARSELT_SPARSITY_50_PERCENT) ==
                 HIPSPARSE_STATUS_NOT_SUPPORTED,
             "old INT8 rejection");
      std::puts("BASELINE_REJECTED_INT8");
      return 0;
    }
    int cases = 0;
    auto check = [&](hipsparseStatus_t got, hipsparseStatus_t wanted,
                     const char *label) {
      expect(got == wanted, label);
      std::printf("PASS integer API %s\n", label);
      ++cases;
    };
    auto init = [&](int64_t k, hipDataType output = HIP_R_8I) {
      descriptor(h, da, layout(5, k, HIPSPARSE_ORDER_ROW, 1), HIP_R_8I, true);
      descriptor(h, db, layout(k, 7, HIPSPARSE_ORDER_ROW, 1), HIP_R_8I, false);
      descriptor(h, dc, layout(5, 7, HIPSPARSE_ORDER_ROW, 1), output, false);
      descriptor(h, dd, layout(5, 7, HIPSPARSE_ORDER_ROW, 1), output, false);
    };
    hipsparseLtMatmulDescriptor_t mm{};
    auto multiply = [&](hipsparseLtComputetype_t compute) {
      return hipsparseLtMatmulDescriptorInit(
          &h, &mm, HIPSPARSE_OPERATION_NON_TRANSPOSE,
          HIPSPARSE_OPERATION_NON_TRANSPOSE, &da, &db, &dc, &dd, compute);
    };
    init(262140);
    check(multiply(HIPSPARSELT_COMPUTE_32I), HIPSPARSE_STATUS_SUCCESS,
          "safe K boundary");
    auto snapshot = mm;
    init(262144);
    check(multiply(HIPSPARSELT_COMPUTE_32I), HIPSPARSE_STATUS_NOT_SUPPORTED,
          "overflow K boundary");
    expect(std::memcmp(&snapshot, &mm, sizeof(mm)) == 0,
           "failed init preserves descriptor");
    init(10);
    check(multiply(HIPSPARSELT_COMPUTE_32I), HIPSPARSE_STATUS_NOT_SUPPORTED,
          "2:4 K alignment");
    for (auto output : {HIP_R_16F, HIP_R_16BF, HIP_R_32F}) {
      init(12, output);
      check(multiply(HIPSPARSELT_COMPUTE_32I), HIPSPARSE_STATUS_NOT_SUPPORTED,
            "unadmitted widened output");
    }
    init(12);
    check(multiply(HIPSPARSELT_COMPUTE_32F), HIPSPARSE_STATUS_NOT_SUPPORTED,
          "wrong compute");
    LT(multiply(HIPSPARSELT_COMPUTE_32I));
    hipDataType bias_type{};
    LT(hipsparseLtMatmulDescGetAttribute(&h, &mm, HIPSPARSELT_MATMUL_BIAS_TYPE,
                                         &bias_type, sizeof(bias_type)));
    expect(bias_type == HIP_R_32F, "INT8 default FP32 bias");
    for (auto type : {HIP_R_8I, HIP_R_16F, HIP_R_16BF, HIP_R_32I})
      check(hipsparseLtMatmulDescSetAttribute(
                &h, &mm, HIPSPARSELT_MATMUL_BIAS_TYPE, &type, sizeof(type)),
            HIPSPARSE_STATUS_NOT_SUPPORTED, "non-FP32 integer bias");
    for (auto attr : {HIPSPARSELT_MATMUL_ACTIVATION_SIGMOID,
                      HIPSPARSELT_MATMUL_ACTIVATION_TANH}) {
      int enabled = 1;
      check(hipsparseLtMatmulDescSetAttribute(&h, &mm, attr, &enabled,
                                              sizeof(enabled)),
            HIPSPARSE_STATUS_NOT_SUPPORTED,
            "upstream unsupported integer activation");
      enabled = 0;
      check(hipsparseLtMatmulDescSetAttribute(&h, &mm, attr, &enabled,
                                              sizeof(enabled)),
            HIPSPARSE_STATUS_SUCCESS, "disable unsupported activation");
    }
    check(hipsparseLtDenseDescriptorInit(&h, &dc, 5, 7, 10, 16, HIP_R_32I,
                                         HIPSPARSE_ORDER_ROW),
          HIPSPARSE_STATUS_NOT_SUPPORTED, "unadmitted INT32 storage");
    const int api_cases = cases;
    for (bool sa : {false, true})
      for (auto order : {HIPSPARSE_ORDER_ROW, HIPSPARSE_ORDER_COL})
        for (bool ta : {false, true})
          for (bool tb : {false, true})
            for (int mask : {0, 1, 2, 4, 7}) {
              run<int8_t>(h, HIP_R_8I, sa, order, ta, tb, 3, false, false, -1,
                          mask);
              std::puts("PASS integer numerical");
              ++cases;
            }
    for (bool sa : {false, true})
      for (auto order : {HIPSPARSE_ORDER_ROW, HIPSPARSE_ORDER_COL})
        for (bool tile : {false, true})
          for (int act : {-1, int(HIPSPARSELT_MATMUL_ACTIVATION_RELU),
                          int(HIPSPARSELT_MATMUL_ACTIVATION_ABS),
                          int(HIPSPARSELT_MATMUL_ACTIVATION_LEAKYRELU),
                          int(HIPSPARSELT_MATMUL_ACTIVATION_GELU)}) {
            run<int8_t>(h, HIP_R_8I, sa, order, true, false, 3, tile, tile, act,
                        3, 4, 0, false, 1);
            std::puts("PASS integer numerical");
            ++cases;
          }
    for (bool sa : {false, true})
      for (int scalars : {1, 2, 3}) {
        run<int8_t>(h, HIP_R_8I, sa, HIPSPARSE_ORDER_ROW, false, true, 1, false,
                    false, -1, 0, 3, scalars, true);
        std::puts("PASS integer numerical");
        ++cases;
      }
    for (bool sa : {false, true}) {
      // Sum=33032192 exceeds exact incremental FP32 precision. Correct INT32
      // accumulation and final scaling+bias give 63.5, rounding to even 64.
      run<int8_t>(h, HIP_R_8I, sa, HIPSPARSE_ORDER_COL, true, true, 1, false,
                  false, -1, 0, 3, 0, false, 0, 1);
      std::puts("PASS integer numerical");
      ++cases;
    }
    for (bool sa : {false, true}) {
      run<int8_t>(h, HIP_R_8I, sa, HIPSPARSE_ORDER_ROW, false, false, 1, false,
                  false, -1, 0, 0, 1, true, 0, 2);
      std::puts("PASS integer numerical");
      ++cases;
      run<int8_t>(h, HIP_R_8I, sa, HIPSPARSE_ORDER_ROW, false, false, 1, false,
                  false, -1, 0, 0, 2, false, 0, 3);
      std::puts("PASS integer numerical");
      ++cases;
      run<int8_t>(h, HIP_R_8I, sa, HIPSPARSE_ORDER_COL, false, true, 3, true,
                  false, -1, 7, 3, 0, false, 2);
      std::puts("PASS integer numerical");
      ++cases;
    }
    LT(hipsparseLtDestroy(&h));
    std::printf("HIPSPARSELT_INTEGER_PASS cases=%d numerical=%d api=%d\n",
                cases, cases - api_cases, api_cases);
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL %s\n", e.what());
    return 1;
  }
}
