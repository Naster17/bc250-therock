// Native zero-stride broadcast qualification; reuses the independent pruning,
// compressed decoding and dense-double reference from the qualified scalar
// probe.
#define main bc250SparseScalarMain
#include "hipsparselt_scalar_smoke.cpp"
#undef main
int main(int argc, char **argv) {
  try {
    hipDeviceProp_t props;
    HIP(hipGetDeviceProperties(&props, 0));
    expect(std::string(props.gcnArchName).substr(0, 7) == "gfx1013",
           "native gfx1013");
    hipsparseLtHandle_t handle{};
    LT(hipsparseLtInit(&handle));
    if (argc > 1 && std::string(argv[1]) == "--baseline") {
      hipsparseLtMatDescriptor_t d{};
      LT(hipsparseLtDenseDescriptorInit(&handle, &d, 4, 4, 4, 16, HIP_R_32F,
                                        HIPSPARSE_ORDER_ROW));
      int64_t zero = 0;
      expect(hipsparseLtMatDescSetAttribute(
                 &handle, &d, HIPSPARSELT_MAT_BATCH_STRIDE, &zero,
                 sizeof(zero)) == HIPSPARSE_STATUS_NOT_SUPPORTED,
             "old zero-stride rejection");
      LT(hipsparseLtMatDescriptorDestroy(&d));
      LT(hipsparseLtDestroy(&handle));
      std::puts("BASELINE_REJECTED_ZERO_STRIDE");
      return 0;
    }
    int cases = 0;
    auto test = [&](auto type, hipDataType dtype, bool sparse_a,
                    hipsparseOrder_t order, bool ta, bool tb, int batches,
                    bool tile, bool partial, int activation, int mask) {
      run<decltype(type)>(handle, dtype, sparse_a, order, ta, tb, batches, tile,
                          partial, activation, mask);
      std::printf("PASS broadcast dtype=%d sparse_a=%d order=%d ta=%d tb=%d "
                  "batches=%d tile=%d mask=%d activation=%d\n",
                  int(dtype), int(sparse_a), int(order), int(ta), int(tb),
                  batches, int(tile), mask, activation);
      ++cases;
    };
    for (bool sparse_a : {false, true})
      for (auto order : {HIPSPARSE_ORDER_ROW, HIPSPARSE_ORDER_COL})
        for (bool ta : {false, true})
          for (bool tb : {false, true})
            for (int mask = 1; mask < 8; ++mask) {
              test(float{}, HIP_R_32F, sparse_a, order, ta, tb, 3, false, false,
                   -1, mask);
              test(__half{}, HIP_R_16F, sparse_a, order, ta, tb, 3, false,
                   false, -1, mask);
              test(hip_bfloat16{}, HIP_R_16BF, sparse_a, order, ta, tb, 3,
                   false, false, -1, mask);
              test(__half{}, HIP_R_16F, sparse_a, order, ta, tb, 3, true, false,
                   -1, mask);
              test(hip_bfloat16{}, HIP_R_16BF, sparse_a, order, ta, tb, 3, true,
                   false, -1, mask);
            }
    for (bool sparse_a : {false, true}) {
      test(float{}, HIP_R_32F, sparse_a, HIPSPARSE_ORDER_ROW, false, false, 1,
           false, false, -1, 7);
      test(__half{}, HIP_R_16F, sparse_a, HIPSPARSE_ORDER_ROW, false, false, 1,
           true, false, -1, 7);
      test(hip_bfloat16{}, HIP_R_16BF, sparse_a, HIPSPARSE_ORDER_ROW, false,
           false, 1, true, false, -1, 7);
    }
    test(__half{}, HIP_R_16F, true, HIPSPARSE_ORDER_COL, true, false, 3, true,
         true, -1, 1);
    test(hip_bfloat16{}, HIP_R_16BF, false, HIPSPARSE_ORDER_ROW, false, true, 3,
         true, true, -1, 2);
    for (auto activation :
         {HIPSPARSELT_MATMUL_ACTIVATION_RELU, HIPSPARSELT_MATMUL_ACTIVATION_ABS,
          HIPSPARSELT_MATMUL_ACTIVATION_LEAKYRELU,
          HIPSPARSELT_MATMUL_ACTIVATION_GELU,
          HIPSPARSELT_MATMUL_ACTIVATION_SIGMOID,
          HIPSPARSELT_MATMUL_ACTIVATION_TANH})
      for (bool sparse_a : {false, true}) {
        test(float{}, HIP_R_32F, sparse_a, HIPSPARSE_ORDER_ROW, false, true, 3,
             false, false, int(activation), 7);
        test(__half{}, HIP_R_16F, sparse_a, HIPSPARSE_ORDER_COL, true, false, 3,
             false, false, int(activation), 7);
        test(hip_bfloat16{}, HIP_R_16BF, sparse_a, HIPSPARSE_ORDER_ROW, true,
             true, 3, false, false, int(activation), 7);
      }
    // Descriptor safety: zero-stride is an input broadcast, never a batched
    // output.
    auto a = layout(4, 8, HIPSPARSE_ORDER_ROW, 3),
         b = layout(8, 4, HIPSPARSE_ORDER_ROW, 3),
         c = layout(4, 4, HIPSPARSE_ORDER_ROW, 3), d = c;
    hipsparseLtMatDescriptor_t da{}, db{}, dc{}, dd{};
    hipsparseLtMatmulDescriptor_t mm{};
    descriptor(handle, da, a, HIP_R_32F, true);
    descriptor(handle, db, b, HIP_R_32F, false);
    descriptor(handle, dc, c, HIP_R_32F, false);
    descriptor(handle, dd, d, HIP_R_32F, false);
    int64_t zero = 0;
    LT(hipsparseLtMatDescSetAttribute(
        &handle, &dd, HIPSPARSELT_MAT_BATCH_STRIDE, &zero, sizeof(zero)));
    expect(hipsparseLtMatmulDescriptorInit(
               &handle, &mm, HIPSPARSE_OPERATION_NON_TRANSPOSE,
               HIPSPARSE_OPERATION_NON_TRANSPOSE, &da, &db, &dc, &dd,
               HIPSPARSELT_COMPUTE_32F) == HIPSPARSE_STATUS_NOT_SUPPORTED,
           "zero-stride output rejects");
    std::puts("PASS reject batched zero-stride D");
    ++cases;
    LT(hipsparseLtMatDescSetAttribute(&handle, &dd,
                                      HIPSPARSELT_MAT_BATCH_STRIDE, &d.stride,
                                      sizeof(d.stride)));
    int two = 2;
    LT(hipsparseLtMatDescSetAttribute(&handle, &db, HIPSPARSELT_MAT_NUM_BATCHES,
                                      &two, sizeof(two)));
    expect(hipsparseLtMatmulDescriptorInit(
               &handle, &mm, HIPSPARSE_OPERATION_NON_TRANSPOSE,
               HIPSPARSE_OPERATION_NON_TRANSPOSE, &da, &db, &dc, &dd,
               HIPSPARSELT_COMPUTE_32F) == HIPSPARSE_STATUS_NOT_SUPPORTED,
           "mismatched batch counts reject");
    std::puts("PASS reject mismatched counts");
    ++cases;
    for (int64_t bad :
         {int64_t(-1), int64_t(1), std::numeric_limits<int64_t>::max()}) {
      expect(hipsparseLtMatDescSetAttribute(
                 &handle, &dc, HIPSPARSELT_MAT_BATCH_STRIDE, &bad,
                 sizeof(bad)) == HIPSPARSE_STATUS_NOT_SUPPORTED,
             "bad stride");
      int64_t observed = 0;
      LT(hipsparseLtMatDescGetAttribute(&handle, &dc,
                                        HIPSPARSELT_MAT_BATCH_STRIDE, &observed,
                                        sizeof(observed)));
      expect(observed == c.stride, "failed setter preserves stride");
      std::puts("PASS reject invalid stride");
      ++cases;
    }
    for (auto desc : {&da, &db, &dc, &dd})
      LT(hipsparseLtMatDescriptorDestroy(desc));
    LT(hipsparseLtDestroy(&handle));
    std::printf("HIPSPARSELT_BROADCAST_PASS cases=%d\n", cases);
  } catch (std::exception const &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}
