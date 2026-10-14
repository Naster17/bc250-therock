// Native device FP32 per-row alpha vector; independent dense-double references.
#define main bc250SparseScalarMain
#include "hipsparselt_scalar_smoke.cpp"
#undef main
int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  try {
    hipDeviceProp_t props;
    HIP(hipGetDeviceProperties(&props, 0));
    expect(std::string(props.gcnArchName).substr(0, 7) == "gfx1013",
           "native gfx1013");
    hipsparseLtHandle_t h{};
    LT(hipsparseLtInit(&h));
    auto a = layout(4, 8, HIPSPARSE_ORDER_ROW, 3),
         b = layout(8, 4, HIPSPARSE_ORDER_ROW, 3),
         c = layout(4, 4, HIPSPARSE_ORDER_ROW, 3);
    hipsparseLtMatDescriptor_t da{}, db{}, dc{}, dd{};
    hipsparseLtMatmulDescriptor_t mm{};
    descriptor(h, da, a, HIP_R_32F, true);
    descriptor(h, db, b, HIP_R_32F, false);
    descriptor(h, dc, c, HIP_R_32F, false);
    descriptor(h, dd, c, HIP_R_32F, false);
    LT(hipsparseLtMatmulDescriptorInit(&h, &mm,
                                       HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                       HIPSPARSE_OPERATION_NON_TRANSPOSE, &da,
                                       &db, &dc, &dd, HIPSPARSELT_COMPUTE_32F));
    int enabled = 1;
    if (argc > 1 && std::string(argv[1]) == "--baseline") {
      expect(hipsparseLtMatmulDescSetAttribute(
                 &h, &mm, HIPSPARSELT_MATMUL_ALPHA_VECTOR_SCALING, &enabled,
                 sizeof(enabled)) == HIPSPARSE_STATUS_NOT_SUPPORTED,
             "old alpha vector rejection");
      std::puts("BASELINE_REJECTED_ALPHA_VECTOR");
      return 0;
    }
    int cases = 0;
    auto check = [&](hipsparseStatus_t got, hipsparseStatus_t wanted,
                     const char *label) {
      expect(got == wanted, label);
      std::printf("PASS vector API %s\n", label);
      ++cases;
    };
    for (auto attr : {HIPSPARSELT_MATMUL_ALPHA_VECTOR_SCALING,
                      HIPSPARSELT_MATMUL_BETA_VECTOR_SCALING}) {
      for (size_t bytes :
           {size_t(0), size_t(1), size_t(3), size_t(8), size_t(4096)}) {
        uint64_t buffer = 0x0101010101010101ULL;
        check(hipsparseLtMatmulDescSetAttribute(&h, &mm, attr, &buffer, bytes),
              HIPSPARSE_STATUS_INVALID_VALUE, "setter size");
        check(hipsparseLtMatmulDescGetAttribute(&h, &mm, attr, &buffer, bytes),
              HIPSPARSE_STATUS_INVALID_VALUE, "getter size");
        expect(buffer == 0x0101010101010101ULL, "getter guard");
      }
      for (int invalid : {-1, 2, 99})
        check(hipsparseLtMatmulDescSetAttribute(&h, &mm, attr, &invalid,
                                                sizeof(invalid)),
              HIPSPARSE_STATUS_INVALID_VALUE, "invalid flag");
      check(hipsparseLtMatmulDescSetAttribute(&h, &mm, attr, nullptr, 4),
            HIPSPARSE_STATUS_INVALID_VALUE, "null setter");
      check(hipsparseLtMatmulDescGetAttribute(&h, &mm, attr, nullptr, 4),
            HIPSPARSE_STATUS_INVALID_VALUE, "null getter");
      int observed = 99;
      LT(hipsparseLtMatmulDescGetAttribute(&h, &mm, attr, &observed,
                                           sizeof(observed)));
      expect(observed == 0, "failed requests preserve flag");
    }
    check(hipsparseLtMatmulDescSetAttribute(
              &h, &mm, HIPSPARSELT_MATMUL_BETA_VECTOR_SCALING, &enabled,
              sizeof(enabled)),
          HIPSPARSE_STATUS_NOT_SUPPORTED, "beta vector unsupported");
    int disabled = 0;
    check(hipsparseLtMatmulDescSetAttribute(
              &h, &mm, HIPSPARSELT_MATMUL_BETA_VECTOR_SCALING, &disabled,
              sizeof(disabled)),
          HIPSPARSE_STATUS_SUCCESS, "scalar beta accepts disable");
    check(hipsparseLtMatmulDescSetAttribute(
              &h, &mm, HIPSPARSELT_MATMUL_ALPHA_VECTOR_SCALING, &enabled,
              sizeof(enabled)),
          HIPSPARSE_STATUS_SUCCESS, "enable alpha vector");
    int observed = 0;
    LT(hipsparseLtMatmulDescGetAttribute(
        &h, &mm, HIPSPARSELT_MATMUL_ALPHA_VECTOR_SCALING, &observed,
        sizeof(observed)));
    expect(observed == 1, "enable roundtrip");
    check(hipsparseLtMatmulDescSetAttribute(
              &h, &mm, HIPSPARSELT_MATMUL_ALPHA_VECTOR_SCALING, &disabled,
              sizeof(disabled)),
          HIPSPARSE_STATUS_SUCCESS, "disable alpha vector");
    observed = 99;
    LT(hipsparseLtMatmulDescGetAttribute(
        &h, &mm, HIPSPARSELT_MATMUL_ALPHA_VECTOR_SCALING, &observed,
        sizeof(observed)));
    expect(observed == 0, "disable alpha vector roundtrip");
    for (auto desc : {&da, &db, &dc, &dd})
      LT(hipsparseLtMatDescriptorDestroy(desc));
    auto test = [&](auto type, hipDataType dtype, bool sparse_a,
                    hipsparseOrder_t order, bool ta, bool tb, int batches,
                    bool tile, bool partial, int activation, int mask, int bias,
                    int mode = 1, int scalar = 0, bool inplace = false) {
      run<decltype(type)>(h, dtype, sparse_a, order, ta, tb, batches, tile,
                          partial, activation, mask, bias, scalar, inplace,
                          mode);
      std::printf("PASS vector GPU dtype=%d sparseA=%d order=%d ta=%d tb=%d "
                  "batch=%d tile=%d mask=%d bias=%d mode=%d scalar=%d "
                  "inplace=%d activation=%d\n",
                  int(dtype), int(sparse_a), int(order), int(ta), int(tb),
                  batches, int(tile), mask, bias, mode, scalar, int(inplace),
                  activation);
      ++cases;
    };
    for (bool sparse_a : {false, true})
      for (auto order : {HIPSPARSE_ORDER_ROW, HIPSPARSE_ORDER_COL})
        for (bool ta : {false, true})
          for (bool tb : {false, true})
            for (int bias : {0, 1, 4}) {
              test(float{}, HIP_R_32F, sparse_a, order, ta, tb, 3, false, false,
                   -1, 0, bias);
              test(__half{}, HIP_R_16F, sparse_a, order, ta, tb, 3, false,
                   false, -1, 0, bias);
              test(hip_bfloat16{}, HIP_R_16BF, sparse_a, order, ta, tb, 3,
                   false, false, -1, 0, bias);
              test(__half{}, HIP_R_16F, sparse_a, order, ta, tb, 3, true, false,
                   -1, 0, bias);
              test(hip_bfloat16{}, HIP_R_16BF, sparse_a, order, ta, tb, 3, true,
                   false, -1, 0, bias);
            }
    for (bool sparse_a : {false, true})
      for (int mask = 1; mask < 8; ++mask)
        for (int bias : {0, 4}) {
          test(float{}, HIP_R_32F, sparse_a, HIPSPARSE_ORDER_COL, true, false,
               3, false, false, -1, mask, bias);
          test(__half{}, HIP_R_16F, sparse_a, HIPSPARSE_ORDER_ROW, false, true,
               3, false, false, -1, mask, bias);
          test(hip_bfloat16{}, HIP_R_16BF, sparse_a, HIPSPARSE_ORDER_COL, true,
               true, 3, false, false, -1, mask, bias);
        }
    for (auto activation :
         {HIPSPARSELT_MATMUL_ACTIVATION_RELU, HIPSPARSELT_MATMUL_ACTIVATION_ABS,
          HIPSPARSELT_MATMUL_ACTIVATION_LEAKYRELU,
          HIPSPARSELT_MATMUL_ACTIVATION_GELU,
          HIPSPARSELT_MATMUL_ACTIVATION_SIGMOID,
          HIPSPARSELT_MATMUL_ACTIVATION_TANH})
      for (bool sparse_a : {false, true}) {
        test(float{}, HIP_R_32F, sparse_a, HIPSPARSE_ORDER_ROW, false, true, 3,
             false, false, int(activation), 7, 4);
        test(__half{}, HIP_R_16F, sparse_a, HIPSPARSE_ORDER_COL, true, false, 3,
             false, false, int(activation), 7, 1);
        test(hip_bfloat16{}, HIP_R_16BF, sparse_a, HIPSPARSE_ORDER_ROW, true,
             true, 3, false, false, int(activation), 7, 4);
      }
    for (int mode : {1, 2})
      for (int scalar : {0, 2})
        for (int bias : {0, 4}) {
          test(float{}, HIP_R_32F, true, HIPSPARSE_ORDER_COL, true, false, 3,
               false, false, -1, 7, bias, mode, scalar);
          test(__half{}, HIP_R_16F, false, HIPSPARSE_ORDER_ROW, false, true, 3,
               false, false, -1, 7, bias, mode, scalar);
          test(hip_bfloat16{}, HIP_R_16BF, true, HIPSPARSE_ORDER_COL, true,
               true, 3, false, false, -1, 7, bias, mode, scalar);
        }
    for (int batches : {1, 3})
      for (int bias : {0, 1, 4}) {
        test(float{}, HIP_R_32F, true, HIPSPARSE_ORDER_ROW, false, false,
             batches, false, false, -1, 0, bias, 1, 0, true);
        test(__half{}, HIP_R_16F, false, HIPSPARSE_ORDER_COL, true, false,
             batches, false, false, -1, 0, bias, 1, 0, true);
        test(hip_bfloat16{}, HIP_R_16BF, true, HIPSPARSE_ORDER_ROW, false, true,
             batches, false, false, -1, 0, bias, 1, 0, true);
      }
    test(__half{}, HIP_R_16F, true, HIPSPARSE_ORDER_COL, true, false, 3, true,
         true, -1, 1, 4);
    test(hip_bfloat16{}, HIP_R_16BF, false, HIPSPARSE_ORDER_ROW, false, true, 3,
         true, true, -1, 2, 1);
    LT(hipsparseLtDestroy(&h));
    std::printf("HIPSPARSELT_VECTOR_PASS cases=%d\n", cases);
  } catch (std::exception const &e) {
    std::fprintf(stderr, "FAIL %s\n", e.what());
    return 1;
  }
}
