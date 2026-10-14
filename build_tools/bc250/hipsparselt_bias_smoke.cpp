// Native row-bias epilogue: independent dense-double math and compressed
// decoding.
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
    const void *pointer = nullptr;
    if (argc > 1 && std::string(argv[1]) == "--baseline") {
      expect(hipsparseLtMatmulDescSetAttribute(
                 &h, &mm, HIPSPARSELT_MATMUL_BIAS_POINTER, &pointer,
                 sizeof(pointer)) == HIPSPARSE_STATUS_NOT_SUPPORTED,
             "old bias rejection");
      std::puts("BASELINE_REJECTED_BIAS");
      return 0;
    }
    int cases = 0;
    auto check = [&](hipsparseStatus_t got, hipsparseStatus_t expected,
                     const char *label) {
      expect(got == expected, label);
      std::printf("PASS bias API %s\n", label);
      ++cases;
    };
    for (auto attr :
         {HIPSPARSELT_MATMUL_BIAS_POINTER, HIPSPARSELT_MATMUL_BIAS_STRIDE,
          HIPSPARSELT_MATMUL_BIAS_TYPE})
      for (size_t size :
           {size_t(0), size_t(1), size_t(3), size_t(16), size_t(4096)}) {
        uint64_t data = 0x0101010101010101ULL;
        check(hipsparseLtMatmulDescSetAttribute(&h, &mm, attr, &data, size),
              HIPSPARSE_STATUS_INVALID_VALUE, "setter size");
        check(hipsparseLtMatmulDescGetAttribute(&h, &mm, attr, &data, size),
              HIPSPARSE_STATUS_INVALID_VALUE, "getter size");
        expect(data == 0x0101010101010101ULL, "failed getter does not write");
      }
    for (int64_t stride :
         {int64_t(-1), int64_t(1), std::numeric_limits<int64_t>::max()}) {
      check(hipsparseLtMatmulDescSetAttribute(&h, &mm,
                                              HIPSPARSELT_MATMUL_BIAS_STRIDE,
                                              &stride, sizeof(stride)),
            stride == std::numeric_limits<int64_t>::max()
                ? HIPSPARSE_STATUS_NOT_SUPPORTED
                : HIPSPARSE_STATUS_INVALID_VALUE,
            "invalid stride");
      int64_t observed = -1;
      LT(hipsparseLtMatmulDescGetAttribute(&h, &mm,
                                           HIPSPARSELT_MATMUL_BIAS_STRIDE,
                                           &observed, sizeof(observed)));
      expect(observed == 0, "failed stride setter atomic");
    }
    for (auto dtype : {HIP_R_16F, HIP_R_16BF, HIP_R_64F, HIP_R_8I,
                       static_cast<hipDataType>(999)}) {
      check(hipsparseLtMatmulDescSetAttribute(
                &h, &mm, HIPSPARSELT_MATMUL_BIAS_TYPE, &dtype, sizeof(dtype)),
            HIPSPARSE_STATUS_NOT_SUPPORTED, "invalid bias type for FP32");
      hipDataType observed{};
      LT(hipsparseLtMatmulDescGetAttribute(
          &h, &mm, HIPSPARSELT_MATMUL_BIAS_TYPE, &observed, sizeof(observed)));
      expect(observed == HIP_R_32F, "failed type setter atomic");
    }
    Device data(1024);
    pointer = static_cast<unsigned char *>(data.data) + 1;
    check(hipsparseLtMatmulDescSetAttribute(&h, &mm,
                                            HIPSPARSELT_MATMUL_BIAS_POINTER,
                                            &pointer, sizeof(pointer)),
          HIPSPARSE_STATUS_INVALID_VALUE, "unaligned bias");
    const void *observed = reinterpret_cast<const void *>(1);
    LT(hipsparseLtMatmulDescGetAttribute(
        &h, &mm, HIPSPARSELT_MATMUL_BIAS_POINTER, &observed, sizeof(observed)));
    expect(observed == nullptr, "failed pointer setter atomic");
    pointer = data.data;
    LT(hipsparseLtMatmulDescSetAttribute(
        &h, &mm, HIPSPARSELT_MATMUL_BIAS_POINTER, &pointer, sizeof(pointer)));
    hipsparseLtMatmulAlgSelection_t alg{};
    hipsparseLtMatmulPlan_t plan{};
    LT(hipsparseLtMatmulAlgSelectionInit(&h, &alg, &mm,
                                         HIPSPARSELT_MATMUL_ALG_DEFAULT));
    LT(hipsparseLtMatmulPlanInit(&h, &plan, &mm, &alg));
    Device ga(1024), gb(1024), gc(1024);
    float alpha = 1, beta = 1;
    for (int offset : {0, 1})
      check(hipsparseLtMatmul(&h, &plan, &alpha, ga.data, gb.data, &beta,
                              gc.data, static_cast<float *>(data.data) + offset,
                              nullptr, nullptr, 0),
            HIPSPARSE_STATUS_NOT_SUPPORTED, "D overlaps bias span");
    LT(hipsparseLtMatmulPlanDestroy(&plan));
    LT(hipsparseLtMatmulAlgSelectionDestroy(&alg));
    for (auto desc : {&da, &db, &dc, &dd})
      LT(hipsparseLtMatDescriptorDestroy(desc));
    auto test = [&](auto type, hipDataType dtype, bool sparse_a,
                    hipsparseOrder_t order, bool ta, bool tb, int batches,
                    bool tile, bool partial, int activation, int mask, int bias,
                    int scalar = 0, bool inplace = false) {
      run<decltype(type)>(h, dtype, sparse_a, order, ta, tb, batches, tile,
                          partial, activation, mask, bias, scalar, inplace);
      std::printf(
          "PASS bias GPU dtype=%d sparseA=%d order=%d ta=%d tb=%d batch=%d "
          "tile=%d bias=%d mask=%d activation=%d scalar=%d inplace=%d\n",
          int(dtype), int(sparse_a), int(order), int(ta), int(tb), batches,
          int(tile), bias, mask, activation, scalar, int(inplace));
      ++cases;
    };
    for (bool sparse_a : {false, true})
      for (auto order : {HIPSPARSE_ORDER_ROW, HIPSPARSE_ORDER_COL})
        for (bool ta : {false, true})
          for (bool tb : {false, true})
            for (int bias : {1, 2, 3, 4}) {
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
      for (int mask : {1, 2, 4, 7})
        for (int bias : {1, 2, 3, 4}) {
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
      for (bool sparse_a : {false, true})
        for (int bias : {1, 4}) {
          test(float{}, HIP_R_32F, sparse_a, HIPSPARSE_ORDER_ROW, false, true,
               3, false, false, int(activation), 7, bias);
          test(__half{}, HIP_R_16F, sparse_a, HIPSPARSE_ORDER_COL, true, false,
               3, false, false, int(activation), 7, bias);
          test(hip_bfloat16{}, HIP_R_16BF, sparse_a, HIPSPARSE_ORDER_ROW, true,
               true, 3, false, false, int(activation), 7, bias);
        }
    for (int scalar : {1, 2, 3})
      for (int bias : {1, 4}) {
        test(float{}, HIP_R_32F, true, HIPSPARSE_ORDER_COL, true, false, 3,
             false, false, -1, 7, bias, scalar);
        test(__half{}, HIP_R_16F, false, HIPSPARSE_ORDER_ROW, false, true, 3,
             false, false, -1, 7, bias, scalar);
        test(hip_bfloat16{}, HIP_R_16BF, true, HIPSPARSE_ORDER_COL, true, true,
             3, false, false, -1, 7, bias, scalar);
      }
    for (int bias : {1, 4, 5})
      for (int batches : {1, 3}) {
        test(float{}, HIP_R_32F, true, HIPSPARSE_ORDER_ROW, false, false,
             batches, false, false, -1, 0, bias, 0, true);
        test(__half{}, HIP_R_16F, false, HIPSPARSE_ORDER_COL, true, false,
             batches, false, false, -1, 0, bias, 0, true);
        test(hip_bfloat16{}, HIP_R_16BF, true, HIPSPARSE_ORDER_ROW, false, true,
             batches, false, false, -1, 0, bias, 0, true);
      }
    test(__half{}, HIP_R_16F, true, HIPSPARSE_ORDER_COL, true, false, 3, true,
         true, -1, 1, 4);
    test(hip_bfloat16{}, HIP_R_16BF, false, HIPSPARSE_ORDER_ROW, false, true, 3,
         true, true, -1, 2, 2);
    LT(hipsparseLtDestroy(&h));
    std::printf("HIPSPARSELT_BIAS_PASS cases=%d\n", cases);
  } catch (std::exception const &e) {
    std::fprintf(stderr, "FAIL %s\n", e.what());
    return 1;
  }
}
