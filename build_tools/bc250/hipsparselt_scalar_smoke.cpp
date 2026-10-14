// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Public API qualification of the native sparse scalar backend against dense
// double-precision CPU GEMM, plus independent pruning and compressed decoding.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#include <hipsparselt/hipsparselt.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#define HIP(call)                                                              \
  do {                                                                         \
    auto status = (call);                                                      \
    if (status != hipSuccess)                                                  \
      throw std::runtime_error(std::string(#call) +                            \
                               " HIP=" + hipGetErrorName(status));             \
  } while (0)
#define LT(call)                                                               \
  do {                                                                         \
    auto status = (call);                                                      \
    if (status != HIPSPARSE_STATUS_SUCCESS)                                    \
      throw std::runtime_error(std::string(#call) +                            \
                               " status=" + std::to_string(int(status)));      \
  } while (0)
void expect(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
void delayed_marker(void *data) {
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  static_cast<std::atomic<bool> *>(data)->store(true);
}
struct Device {
  void *data = nullptr;
  explicit Device(size_t bytes) { HIP(hipMalloc(&data, bytes)); }
  ~Device() {
    if (data)
      (void)hipFree(data);
  }
  Device(const Device &) = delete;
};
struct Layout {
  int64_t rows, cols, ld, stride;
  hipsparseOrder_t order;
  int batches;
  size_t storage() const {
    return stride == 0 ? ld * (order == HIPSPARSE_ORDER_ROW ? rows : cols)
                       : stride * batches;
  }
  size_t index(int batch, int64_t row, int64_t col, bool trans = false) const {
    if (trans)
      std::swap(row, col);
    return batch * stride +
           (order == HIPSPARSE_ORDER_ROW ? row * ld + col : col * ld + row);
  }
};
Layout layout(int64_t r, int64_t c, hipsparseOrder_t order, int batches) {
  int64_t ld = (order == HIPSPARSE_ORDER_ROW ? c : r) + 3;
  return {r,     c,      ld, ld * (order == HIPSPARSE_ORDER_ROW ? r : c) + 5,
          order, batches};
}
void descriptor(hipsparseLtHandle_t &handle, hipsparseLtMatDescriptor_t &desc,
                const Layout &l, hipDataType type, bool sparse) {
  if (sparse)
    LT(hipsparseLtStructuredDescriptorInit(&handle, &desc, l.rows, l.cols, l.ld,
                                           16, type, l.order,
                                           HIPSPARSELT_SPARSITY_50_PERCENT));
  else
    LT(hipsparseLtDenseDescriptorInit(&handle, &desc, l.rows, l.cols, l.ld, 16,
                                      type, l.order));
  LT(hipsparseLtMatDescSetAttribute(&handle, &desc, HIPSPARSELT_MAT_NUM_BATCHES,
                                    &l.batches, sizeof(l.batches)));
  LT(hipsparseLtMatDescSetAttribute(&handle, &desc,
                                    HIPSPARSELT_MAT_BATCH_STRIDE, &l.stride,
                                    sizeof(l.stride)));
}

const std::vector<unsigned> &tile_masks() {
  static const std::vector<unsigned> masks = [] {
    std::vector<unsigned> result;
    // Independent exhaustive 16-bit search, rather than the GPU's three-row
    // enumeration and derived fourth row.
    for (unsigned mask = 0; mask < 65536; ++mask) {
      if (__builtin_popcount(mask) != 8)
        continue;
      bool valid = true;
      for (int r = 0; r < 4; ++r)
        valid &= __builtin_popcount((mask >> (r * 4)) & 15) == 2;
      for (int c = 0; c < 4; ++c) {
        int count = 0;
        for (int r = 0; r < 4; ++r)
          count += (mask >> (r * 4 + c)) & 1;
        valid &= count == 2;
      }
      if (valid)
        result.push_back(mask);
    }
    expect(result.size() == 90, "exactly 90 doubly constrained tile masks");
    return result;
  }();
  return masks;
}

template <class T>
void run(hipsparseLtHandle_t &handle, hipDataType type, bool sparse_a,
         hipsparseOrder_t order, bool ta, bool tb, int batches,
         bool tile_mode = false, bool partial_tile = false, int activation = -1,
         int broadcast_mask = 0, int bias_mode = 0, int scalar_mode = 0,
         bool inplace_cd = false) {
  std::fprintf(stderr,
               "CASE type=%d sparse_a=%d order=%d ta=%d tb=%d batches=%d "
               "tile=%d partial=%d\n",
               int(type), int(sparse_a), int(order), int(ta), int(tb), batches,
               int(tile_mode), int(partial_tile));
  const int64_t m = tile_mode && !partial_tile ? 8 : 5;
  const int64_t n = tile_mode && !partial_tile ? 12 : 7;
  constexpr int64_t k = 12;
  auto prune_alg =
      tile_mode ? HIPSPARSELT_PRUNE_SPMMA_TILE : HIPSPARSELT_PRUNE_SPMMA_STRIP;
  const int group = type == HIP_R_32F ? 2 : 4, kept = group / 2;
  auto la = layout(ta ? k : m, ta ? m : k, order, batches);
  auto lb = layout(tb ? n : k, tb ? k : n, order, batches);
  auto lc = layout(m, n, order, batches);
  auto ld = layout(m, n,
                   order == HIPSPARSE_ORDER_ROW ? HIPSPARSE_ORDER_COL
                                                : HIPSPARSE_ORDER_ROW,
                   batches);
  if (inplace_cd)
    lc = ld;
  if (broadcast_mask & 1)
    la.stride = 0;
  if (broadcast_mask & 2)
    lb.stride = 0;
  if (broadcast_mask & 4)
    lc.stride = 0;
  const int sparse_batches = (sparse_a ? la : lb).stride == 0 ? 1 : batches;
  std::vector<T> a(la.storage(), T(-99.0f)), b(lb.storage(), T(-99.0f)),
      c(lc.storage(), T(-99.0f)), output(ld.storage(), T(-99.0f));
  for (int batch = 0; batch < batches; ++batch) {
    for (int row = 0; row < m; ++row)
      for (int col = 0; col < k; ++col)
        a[la.index(batch, row, col, ta)] =
            T(float(((row * 7 + col * 3 + batch * 11) % 19) - 9) * 0.125f);
    for (int row = 0; row < k; ++row)
      for (int col = 0; col < n; ++col)
        b[lb.index(batch, row, col, tb)] =
            T(float(((row * 5 + col * 13 + batch * 3) % 23) - 11) * 0.125f);
    for (int row = 0; row < m; ++row)
      for (int col = 0; col < n; ++col)
        c[lc.index(batch, row, col)] = T(float(row - col + batch) * 0.25f);
  }
  auto &input = sparse_a ? a : b;
  const auto &ls = sparse_a ? la : lb;
  bool transpose = sparse_a ? ta : tb;
  // Include all-zero and one-nonzero groups, highest-index-only groups, and
  // ties.
  for (int batch = 0; batch < batches; ++batch)
    for (int outer = 0; outer < (sparse_a ? m : n); ++outer)
      for (int start = 0; start < k; start += group)
        if ((outer + start + batch) % 3 == 0)
          for (int i = 0; i < group; ++i)
            input[ls.index(batch, sparse_a ? outer : start + i,
                           sparse_a ? start + i : outer, transpose)] =
                T(i == group - 1 ? 0.5f : 0.0f);
  std::vector<T> pruned(input.size(), T(-99.0f)), reference = pruned;
  if (tile_mode) {
    // All rows independently prefer the same two columns: a STRIP alias would
    // violate the TILE column constraint. A second tile contains exact ties.
    for (int batch = 0; batch < batches; ++batch)
      for (int r = 0; r < 4; ++r)
        for (int col = 0; col < 8; ++col)
          input[ls.index(batch, sparse_a ? r : col, sparse_a ? col : r,
                         transpose)] =
              T(col >= 4  ? 1.0f
                : col < 2 ? 8.0f - col
                          : float(2 + r + col) * 0.125f);
    int length = sparse_a ? m : n;
    for (int batch = 0; batch < batches; ++batch)
      for (int first_row = 0; first_row < length; first_row += 4)
        for (int first_col = 0; first_col < k; first_col += 4) {
          double magnitudes[16]{};
          for (int r = 0; r < 4 && first_row + r < length; ++r)
            for (int col = 0; col < 4; ++col)
              magnitudes[r * 4 + col] = std::abs(double(float(input[ls.index(
                  batch, sparse_a ? first_row + r : first_col + col,
                  sparse_a ? first_col + col : first_row + r, transpose)])));
          unsigned selected = 0;
          double best = -1;
          for (unsigned mask : tile_masks()) {
            double score = 0;
            for (int i = 0; i < 16; ++i)
              if ((mask >> i) & 1)
                score += magnitudes[i];
            if (score > best) {
              best = score;
              selected = mask;
            }
          }
          for (int r = 0; r < 4 && first_row + r < length; ++r)
            for (int col = 0; col < 4; ++col) {
              size_t position = ls.index(
                  batch, sparse_a ? first_row + r : first_col + col,
                  sparse_a ? first_col + col : first_row + r, transpose);
              reference[position] =
                  ((selected >> (r * 4 + col)) & 1) ? input[position] : T(0.0f);
            }
        }
  } else {
    for (int batch = 0; batch < batches; ++batch)
      for (int outer = 0; outer < (sparse_a ? m : n); ++outer)
        for (int start = 0; start < k; start += group) {
          int indices[4] = {0, 1, 2, 3};
          auto index = [&](int i) {
            return ls.index(batch, sparse_a ? outer : start + i,
                            sparse_a ? start + i : outer, transpose);
          };
          std::stable_sort(indices, indices + group, [&](int i, int j) {
            return std::abs(float(input[index(i)])) >
                   std::abs(float(input[index(j)]));
          });
          for (int i = 0; i < group; ++i) {
            bool retain = false;
            for (int j = 0; j < kept; ++j)
              retain |= indices[j] == i;
            reference[index(i)] = retain ? input[index(i)] : T(0.0f);
          }
        }
  }
  hipsparseLtMatDescriptor_t da{}, db{}, dc{}, dd{};
  descriptor(handle, da, la, type, sparse_a);
  descriptor(handle, db, lb, type, !sparse_a);
  descriptor(handle, dc, lc, type, false);
  descriptor(handle, dd, ld, type, false);
  auto opa =
      ta ? HIPSPARSE_OPERATION_TRANSPOSE : HIPSPARSE_OPERATION_NON_TRANSPOSE;
  auto opb =
      tb ? HIPSPARSE_OPERATION_TRANSPOSE : HIPSPARSE_OPERATION_NON_TRANSPOSE;
  const bool float_bias = bias_mode == 3 || bias_mode == 4;
  int64_t bias_stride = bias_mode == 2 || bias_mode == 4 ? m + 3 : 0;
  size_t bias_elements = m + (batches - 1) * bias_stride + 8;
  std::vector<float> host_bias(bias_elements, -17.0f);
  std::vector<T> typed_bias(bias_elements, T(-17.0f));
  for (int batch = 0; batch < (bias_stride ? batches : 1); ++batch)
    for (int row = 0; row < m; ++row) {
      auto index = 4 + batch * bias_stride + row;
      host_bias[index] = float(row - 2 + batch * 3) * 0.125f + 0.00390625f;
      typed_bias[index] = T(host_bias[index]);
    }
  std::unique_ptr<Device> gpu_bias;
  const void *bias_pointer = nullptr;
  if (bias_mode) {
    size_t element_size = float_bias ? sizeof(float) : sizeof(T);
    gpu_bias = std::make_unique<Device>(bias_elements * element_size);
    HIP(hipMemcpy(gpu_bias->data,
                  float_bias ? static_cast<const void *>(host_bias.data())
                             : static_cast<const void *>(typed_bias.data()),
                  bias_elements * element_size, hipMemcpyHostToDevice));
    bias_pointer =
        static_cast<const unsigned char *>(gpu_bias->data) + 4 * element_size;
  }
  hipsparseLtMatmulDescriptor_t mm{};
  hipsparseLtMatmulAlgSelection_t alg{};
  hipsparseLtMatmulPlan_t plan{};
  LT(hipsparseLtMatmulDescriptorInit(&handle, &mm, opa, opb, &da, &db, &dc, &dd,
                                     HIPSPARSELT_COMPUTE_32F));
  if (bias_mode) {
    LT(hipsparseLtMatmulDescSetAttribute(&handle, &mm,
                                         HIPSPARSELT_MATMUL_BIAS_POINTER,
                                         &bias_pointer, sizeof(bias_pointer)));
    LT(hipsparseLtMatmulDescSetAttribute(&handle, &mm,
                                         HIPSPARSELT_MATMUL_BIAS_STRIDE,
                                         &bias_stride, sizeof(bias_stride)));
    if (float_bias) {
      hipDataType bias_type = HIP_R_32F;
      LT(hipsparseLtMatmulDescSetAttribute(&handle, &mm,
                                           HIPSPARSELT_MATMUL_BIAS_TYPE,
                                           &bias_type, sizeof(bias_type)));
    }
    if (bias_mode == 5) {
      bias_pointer = nullptr;
      LT(hipsparseLtMatmulDescSetAttribute(
          &handle, &mm, HIPSPARSELT_MATMUL_BIAS_POINTER, &bias_pointer,
          sizeof(bias_pointer)));
    }
    const void *observed = nullptr;
    int64_t observed_stride = -1;
    hipDataType observed_type{};
    LT(hipsparseLtMatmulDescGetAttribute(&handle, &mm,
                                         HIPSPARSELT_MATMUL_BIAS_POINTER,
                                         &observed, sizeof(observed)));
    LT(hipsparseLtMatmulDescGetAttribute(
        &handle, &mm, HIPSPARSELT_MATMUL_BIAS_STRIDE, &observed_stride,
        sizeof(observed_stride)));
    LT(hipsparseLtMatmulDescGetAttribute(
        &handle, &mm, HIPSPARSELT_MATMUL_BIAS_TYPE, &observed_type,
        sizeof(observed_type)));
    expect(observed == bias_pointer && observed_stride == bias_stride &&
               observed_type == (float_bias ? HIP_R_32F : type),
           "bias attribute roundtrip");
  }
  if (activation >= 0) {
    int enable = 1;
    auto attr = static_cast<hipsparseLtMatmulDescAttribute_t>(activation);
    LT(hipsparseLtMatmulDescSetAttribute(&handle, &mm, attr, &enable,
                                         sizeof(enable)));
    int observed = 0;
    LT(hipsparseLtMatmulDescGetAttribute(&handle, &mm, attr, &observed,
                                         sizeof(observed)));
    expect(observed == 1, "activation attribute roundtrip");
    auto parameter = [&](hipsparseLtMatmulDescAttribute_t attribute,
                         float value) {
      LT(hipsparseLtMatmulDescSetAttribute(&handle, &mm, attribute, &value,
                                           sizeof(value)));
      float observed = 0;
      LT(hipsparseLtMatmulDescGetAttribute(&handle, &mm, attribute, &observed,
                                           sizeof(observed)));
      expect(observed == value, "activation parameter roundtrip");
    };
    if (activation == HIPSPARSELT_MATMUL_ACTIVATION_RELU) {
      parameter(HIPSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD, 0.25f);
      parameter(HIPSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND, 0.75f);
    }
    if (activation == HIPSPARSELT_MATMUL_ACTIVATION_TANH) {
      parameter(HIPSPARSELT_MATMUL_ACTIVATION_TANH_ALPHA, 0.5f);
      parameter(HIPSPARSELT_MATMUL_ACTIVATION_TANH_BETA, 1.25f);
    }
    if (activation == HIPSPARSELT_MATMUL_ACTIVATION_GELU) {
      // The scaling attribute itself enables GELU, as the public HIP backend
      // does.
      int disable = 0;
      LT(hipsparseLtMatmulDescSetAttribute(&handle, &mm, attr, &disable,
                                           sizeof(disable)));
      parameter(HIPSPARSELT_MATMUL_ACTIVATION_GELU_SCALING, 0.75f);
      int enabled = 0;
      LT(hipsparseLtMatmulDescGetAttribute(&handle, &mm, attr, &enabled,
                                           sizeof(enabled)));
      expect(enabled == 1, "GELU scaling enables activation");
    }
    float slope = 0.125f;
    LT(hipsparseLtMatmulDescSetAttribute(
        &handle, &mm, HIPSPARSELT_MATMUL_ACTIVATION_LEAKYRELU_ALPHA, &slope,
        sizeof(slope)));
    float observedSlope = 0;
    LT(hipsparseLtMatmulDescGetAttribute(
        &handle, &mm, HIPSPARSELT_MATMUL_ACTIVATION_LEAKYRELU_ALPHA,
        &observedSlope, sizeof(observedSlope)));
    expect(observedSlope == slope, "slope roundtrip");
  }
  LT(hipsparseLtMatmulAlgSelectionInit(&handle, &alg, &mm,
                                       HIPSPARSELT_MATMUL_ALG_DEFAULT));
  LT(hipsparseLtMatmulPlanInit(&handle, &plan, &mm, &alg));
  if (bias_mode && bias_mode != 5) {
    const void *cleared = nullptr;
    LT(hipsparseLtMatmulDescSetAttribute(&handle, &mm,
                                         HIPSPARSELT_MATMUL_BIAS_POINTER,
                                         &cleared, sizeof(cleared)));
    // Poison every descriptor bias field after creating the owning plan
    // snapshot.
    int64_t changed_stride = m + 9;
    LT(hipsparseLtMatmulDescSetAttribute(
        &handle, &mm, HIPSPARSELT_MATMUL_BIAS_STRIDE, &changed_stride,
        sizeof(changed_stride)));
    hipDataType changed_type = float_bias ? type : HIP_R_32F;
    LT(hipsparseLtMatmulDescSetAttribute(&handle, &mm,
                                         HIPSPARSELT_MATMUL_BIAS_TYPE,
                                         &changed_type, sizeof(changed_type)));
  }
  if (activation >= 0) {
    // The plan owns a snapshot: later descriptor mutations must not change it.
    int disable = 0;
    LT(hipsparseLtMatmulDescSetAttribute(
        &handle, &mm, static_cast<hipsparseLtMatmulDescAttribute_t>(activation),
        &disable, sizeof(disable)));
    int observed = 1;
    LT(hipsparseLtMatmulDescGetAttribute(
        &handle, &mm, static_cast<hipsparseLtMatmulDescAttribute_t>(activation),
        &observed, sizeof(observed)));
    expect(observed == 0, "descriptor mutation after plan creation");
  }
  int max_id = 0;
  LT(hipsparseLtMatmulAlgGetAttribute(&handle, &alg,
                                      HIPSPARSELT_MATMUL_ALG_CONFIG_MAX_ID,
                                      &max_id, sizeof(max_id)));
  expect(max_id == 1, "one real scalar algorithm");
  int unsupported = 2;
  expect(hipsparseLtMatmulAlgSetAttribute(
             &handle, &alg, HIPSPARSELT_MATMUL_SPLIT_K, &unsupported,
             sizeof(unsupported)) == HIPSPARSE_STATUS_NOT_SUPPORTED,
         "split K rejection");
  expect(hipsparseLtMatmulDescSetAttribute(
             &handle, &mm, HIPSPARSELT_MATMUL_ACTIVATION_RELU, &unsupported,
             sizeof(unsupported)) == HIPSPARSE_STATUS_INVALID_VALUE,
         "invalid activation boolean rejection");
  size_t size = 0, workspace = 123;
  LT(hipsparseLtMatmulGetWorkspace(&handle, &plan, &workspace));
  expect(workspace == 0, "zero workspace");
  LT(hipsparseLtSpMMACompressedSize(&handle, &plan, &size, &workspace));
  size_t elements = (sparse_a ? m : n) * k * sparse_batches;
  expect(size == elements / 2 * sizeof(T) + elements / group && workspace == 0,
         "packed size");
  Device gpu_a(a.size() * sizeof(T)), gpu_b(b.size() * sizeof(T)),
      gpu_c(c.size() * sizeof(T)), gpu_d(output.size() * sizeof(T)),
      gpu_sparse(pruned.size() * sizeof(T)), compressed(size + 64),
      invalid(sizeof(int));
  HIP(hipMemcpy(gpu_a.data, a.data(), a.size() * sizeof(T),
                hipMemcpyHostToDevice));
  HIP(hipMemcpy(gpu_b.data, b.data(), b.size() * sizeof(T),
                hipMemcpyHostToDevice));
  HIP(hipMemcpy(gpu_c.data, c.data(), c.size() * sizeof(T),
                hipMemcpyHostToDevice));
  HIP(hipMemcpy(gpu_d.data, output.data(), output.size() * sizeof(T),
                hipMemcpyHostToDevice));
  HIP(hipMemcpy(gpu_sparse.data, pruned.data(), pruned.size() * sizeof(T),
                hipMemcpyHostToDevice));
  HIP(hipMemset(compressed.data, 0xA5, size + 64));
  hipStream_t stream;
  HIP(hipStreamCreate(&stream));
  if (type == HIP_R_32F)
    expect(hipsparseLtSpMMAPrune(&handle, &mm,
                                 sparse_a ? gpu_a.data : gpu_b.data,
                                 gpu_sparse.data, HIPSPARSELT_PRUNE_SPMMA_TILE,
                                 stream) == HIPSPARSE_STATUS_NOT_SUPPORTED,
           "FP32 tile rejects incompatible 1:2 contract");
  LT(hipsparseLtSpMMAPrune(&handle, &mm, sparse_a ? gpu_a.data : gpu_b.data,
                           gpu_sparse.data, prune_alg, stream));
  LT(hipsparseLtSpMMAPruneCheck(&handle, &mm, gpu_sparse.data,
                                static_cast<int *>(invalid.data), stream));
  HIP(hipStreamSynchronize(stream));
  int valid = -1;
  HIP(hipMemcpy(&valid, invalid.data, sizeof(int), hipMemcpyDeviceToHost));
  expect(valid == 0, "valid pruned matrix");
  HIP(hipMemcpy(pruned.data(), gpu_sparse.data, pruned.size() * sizeof(T),
                hipMemcpyDeviceToHost));
  expect(std::memcmp(pruned.data(), reference.data(),
                     pruned.size() * sizeof(T)) == 0,
         "independent magnitude prune including padding");
  if (tile_mode) {
    int length = sparse_a ? m : n;
    for (int batch = 0; batch < batches; ++batch)
      for (int first_row = 0; first_row < length; first_row += 4)
        for (int first_col = 0; first_col < k; first_col += 4) {
          int counts_row[4]{}, counts_col[4]{};
          for (int r = 0; r < 4 && first_row + r < length; ++r)
            for (int col = 0; col < 4; ++col) {
              bool nonzero =
                  float(pruned[ls.index(
                      batch, sparse_a ? first_row + r : first_col + col,
                      sparse_a ? first_col + col : first_row + r,
                      transpose)]) != 0;
              counts_row[r] += nonzero;
              counts_col[col] += nonzero;
            }
          for (int i = 0; i < 4; ++i) {
            expect(counts_row[i] <= 2 && counts_col[i] <= 2,
                   "tile row AND column sparsity");
            if (first_row == 0 && first_col == 0)
              expect(counts_row[i] == 2 && counts_col[i] == 2,
                     "nonzero 4x4 tile retains exactly two in every row and "
                     "column");
          }
        }
  }
  auto *sparse_desc = sparse_a ? &da : &db;
  void *original_sparse = sparse_a ? gpu_a.data : gpu_b.data;
  LT(hipsparseLtSpMMAPrune2(&handle, sparse_desc, sparse_a,
                            sparse_a ? opa : opb, original_sparse,
                            original_sparse, prune_alg, stream));
  HIP(hipStreamSynchronize(stream));
  std::vector<T> inplace(pruned.size());
  HIP(hipMemcpy(inplace.data(), original_sparse, inplace.size() * sizeof(T),
                hipMemcpyDeviceToHost));
  expect(std::memcmp(inplace.data(), reference.data(),
                     inplace.size() * sizeof(T)) == 0,
         "descriptor API in-place pruning");
  if (broadcast_mask) {
    expect(hipsparseLtSpMMACompress(
               &handle, &plan, gpu_sparse.data,
               static_cast<unsigned char *>(gpu_sparse.data) + sizeof(T),
               nullptr, stream) == HIPSPARSE_STATUS_INVALID_VALUE,
           "partial compression overlap rejected");
  }
  LT(hipsparseLtSpMMACompress(&handle, &plan, gpu_sparse.data, compressed.data,
                              nullptr, stream));
  HIP(hipStreamSynchronize(stream));
  std::vector<unsigned char> packed(size + 64);
  HIP(hipMemcpy(packed.data(), compressed.data, size + 64,
                hipMemcpyDeviceToHost));
  for (size_t i = size; i < size + 64; ++i)
    expect(packed[i] == 0xA5, "compressed boundary guard");
  LT(hipsparseLtSpMMACompress2(&handle, sparse_desc, sparse_a,
                               sparse_a ? opa : opb, original_sparse,
                               compressed.data, nullptr, stream));
  HIP(hipStreamSynchronize(stream));
  std::vector<unsigned char> packed2(size + 64);
  HIP(hipMemcpy(packed2.data(), compressed.data, size + 64,
                hipMemcpyDeviceToHost));
  expect(packed == packed2, "descriptor and plan compression agree");
  size_t byte_offset = elements / 2 * sizeof(T);
  for (int batch = 0; batch < sparse_batches; ++batch)
    for (int outer = 0; outer < (sparse_a ? m : n); ++outer)
      for (int start = 0; start < k; start += group) {
        size_t g =
            (batch * (sparse_a ? m : n) + outer) * (k / group) + start / group;
        float decoded[4] = {0, 0, 0, 0};
        unsigned char md = packed[byte_offset + g];
        int previous = -1;
        for (int t = 0; t < kept; ++t) {
          int index = (md >> (2 * t)) & 3;
          expect(index < group && index > previous, "ordered distinct indices");
          previous = index;
          T value;
          std::memcpy(&value, packed.data() + (g * kept + t) * sizeof(T),
                      sizeof(T));
          decoded[index] = float(value);
        }
        for (int i = 0; i < group; ++i)
          expect(decoded[i] == float(reference[ls.index(
                                   batch, sparse_a ? outer : start + i,
                                   sparse_a ? start + i : outer, transpose)]),
                 "compressed data decodes exactly");
      }
  // The original sparse matrix is destroyed before GEMM. This catches any
  // implementation that accepts compressed buffers but secretly uses dense
  // input.
  HIP(hipMemset(sparse_a ? gpu_a.data : gpu_b.data, 0xFF,
                input.size() * sizeof(T)));
  HIP(hipMemset(gpu_sparse.data, 0xFF, pruned.size() * sizeof(T)));
  float alpha = 1.25f, beta = -0.375f;
  if (scalar_mode & 1) {
    alpha = 0;
    HIP(hipMemset(compressed.data, 0xFF, size));
    HIP(hipMemset(sparse_a ? gpu_b.data : gpu_a.data, 0xFF,
                  (sparse_a ? b.size() : a.size()) * sizeof(T)));
  }
  if (scalar_mode & 2) {
    beta = 0;
    HIP(hipMemset(gpu_c.data, 0xFF, c.size() * sizeof(T)));
  }
  if (inplace_cd)
    HIP(hipMemcpy(gpu_d.data, c.data(), c.size() * sizeof(T),
                  hipMemcpyHostToDevice));

  if (broadcast_mask) {
    const void *pa = sparse_a ? compressed.data : gpu_a.data;
    const void *pb = sparse_a ? gpu_b.data : compressed.data;
    for (const void *input : {pa, pb})
      expect(hipsparseLtMatmul(
                 &handle, &plan, &alpha, pa, pb, &beta, gpu_c.data,
                 const_cast<unsigned char *>(
                     static_cast<const unsigned char *>(input)) +
                     sizeof(T),
                 nullptr, &stream, 1) == HIPSPARSE_STATUS_INVALID_VALUE,
             "partial input/output overlap rejected");
    expect(hipsparseLtMatmul(
               &handle, &plan, &alpha, pa, pb, &beta,
               static_cast<unsigned char *>(gpu_d.data) + sizeof(T), gpu_d.data,
               nullptr, &stream, 1) == HIPSPARSE_STATUS_NOT_SUPPORTED,
           "partial C/D overlap rejected");
    if (lc.stride == 0)
      expect(hipsparseLtMatmul(&handle, &plan, &alpha, pa, pb, &beta,
                               gpu_c.data, gpu_c.data, nullptr, &stream,
                               1) == HIPSPARSE_STATUS_NOT_SUPPORTED,
             "broadcast C cannot alias batched D");
  }
  LT(hipsparseLtMatmul(
      &handle, &plan, &alpha, sparse_a ? compressed.data : gpu_a.data,
      sparse_a ? gpu_b.data : compressed.data, &beta,
      inplace_cd ? gpu_d.data : gpu_c.data, gpu_d.data, nullptr, &stream, 1));
  HIP(hipStreamSynchronize(stream));
  HIP(hipMemcpy(output.data(), gpu_d.data, output.size() * sizeof(T),
                hipMemcpyDeviceToHost));
  std::vector<T> expected(output.size(), T(-99.0f));
  for (int batch = 0; batch < batches; ++batch)
    for (int row = 0; row < m; ++row)
      for (int col = 0; col < n; ++col) {
        double sum = 0;
        for (int kk = 0; kk < k; ++kk)
          sum += double(float((sparse_a ? reference
                                        : a)[la.index(batch, row, kk, ta)])) *
                 double(float(
                     (sparse_a ? b : reference)[lb.index(batch, kk, col, tb)]));
        float value =
            float(alpha * sum + beta * float(c[lc.index(batch, row, col)]));
        if (bias_mode && bias_mode != 5) {
          auto index = 4 + batch * bias_stride + row;
          value += float_bias ? host_bias[index] : float(typed_bias[index]);
        }
        if (activation == HIPSPARSELT_MATMUL_ACTIVATION_RELU)
          value = value > 0.25f ? std::min(value, 0.75f) : 0.0f;
        if (activation == HIPSPARSELT_MATMUL_ACTIVATION_ABS)
          value = std::abs(value);
        if (activation == HIPSPARSELT_MATMUL_ACTIVATION_LEAKYRELU && value < 0)
          value *= 0.125f;
        if (activation == HIPSPARSELT_MATMUL_ACTIVATION_GELU)
          value = float(0.75 * 0.5 * double(value) *
                        (1.0 + std::tanh(0.7978845608028654 * double(value) *
                                         (1.0 + 0.044715 * double(value) *
                                                    double(value)))));
        if (activation == HIPSPARSELT_MATMUL_ACTIVATION_SIGMOID)
          value = float(1.0 / (1.0 + std::exp(-double(value))));
        if (activation == HIPSPARSELT_MATMUL_ACTIVATION_TANH)
          value = float(std::tanh(double(value) * 0.5) * 1.25);
        expected[ld.index(batch, row, col)] = T(value);
      }
  // Inputs are powers of two with small K, so FP32 accumulation and CPU double
  // reference are exact; any conversion difference or padding write is a
  // failure.
  bool nonlinear = activation == HIPSPARSELT_MATMUL_ACTIVATION_GELU ||
                   activation == HIPSPARSELT_MATMUL_ACTIVATION_SIGMOID ||
                   activation == HIPSPARSELT_MATMUL_ACTIVATION_TANH;
  if (!nonlinear)
    expect(std::memcmp(expected.data(), output.data(),
                       output.size() * sizeof(T)) == 0,
           "CPU dense double reference and D padding");
  else {
    // Transcendentals are checked against independent double math; storage
    // rounding is bounded by approximately one low-precision ULP. Padding stays
    // exact.
    double tolerance = type == HIP_R_32F   ? 1e-5
                       : type == HIP_R_16F ? 0.001
                                           : 0.008;
    std::vector<bool> written(output.size(), false);
    for (int batch = 0; batch < batches; ++batch)
      for (int row = 0; row < m; ++row)
        for (int col = 0; col < n; ++col)
          written[ld.index(batch, row, col)] = true;
    for (size_t i = 0; i < output.size(); ++i) {
      double got = float(output[i]), ref = float(expected[i]);
      expect(std::isfinite(got) &&
                 (written[i]
                      ? std::abs(got - ref) <= tolerance * (1 + std::abs(ref))
                      : got == ref),
             "nonlinear CPU-double reference and exact padding");
    }
  }
  if (bias_mode) {
    size_t element_size = float_bias ? sizeof(float) : sizeof(T);
    std::vector<unsigned char> observed(bias_elements * element_size);
    HIP(hipMemcpy(observed.data(), gpu_bias->data, observed.size(),
                  hipMemcpyDeviceToHost));
    expect(std::memcmp(observed.data(),
                       float_bias
                           ? static_cast<const void *>(host_bias.data())
                           : static_cast<const void *>(typed_bias.data()),
                       observed.size()) == 0,
           "bias input and boundary guards unchanged");
  }
  if (bias_mode == 0 && activation < 0 && sparse_a &&
      order == HIPSPARSE_ORDER_ROW && !ta && !tb && batches == 1) {
    int repetitions = 2;
    expect(hipsparseLtMatmulAlgSetAttribute(
               &handle, &alg, HIPSPARSELT_MATMUL_SEARCH_ITERATIONS,
               &repetitions,
               sizeof(repetitions)) == HIPSPARSE_STATUS_NOT_SUPPORTED,
           "unsupported search repetitions");
    // alpha=0 ignores NaN operands rather than producing 0*NaN.
    HIP(hipMemset(gpu_b.data, 0xFF, b.size() * sizeof(T)));
    float zero = 0, half = 0.5f;
    LT(hipsparseLtMatmul(&handle, &plan, &zero, compressed.data, gpu_b.data,
                         &half, gpu_c.data, gpu_d.data, nullptr, &stream, 1));
    HIP(hipStreamSynchronize(stream));
    HIP(hipMemcpy(output.data(), gpu_d.data, output.size() * sizeof(T),
                  hipMemcpyDeviceToHost));
    for (int row = 0; row < m; ++row)
      for (int col = 0; col < n; ++col)
        expect(float(output[ld.index(0, row, col)]) ==
                   float(T(half * float(c[lc.index(0, row, col)]))),
               "alpha zero ignores NaN input");
    HIP(hipMemcpy(gpu_b.data, b.data(), b.size() * sizeof(T),
                  hipMemcpyHostToDevice));
    // beta=0 ignores NaN C.
    HIP(hipMemset(gpu_c.data, 0xFF, c.size() * sizeof(T)));
    LT(hipsparseLtMatmul(&handle, &plan, &alpha, compressed.data, gpu_b.data,
                         &zero, gpu_c.data, gpu_d.data, nullptr, &stream, 1));
    HIP(hipStreamSynchronize(stream));
    HIP(hipMemcpy(output.data(), gpu_d.data, output.size() * sizeof(T),
                  hipMemcpyDeviceToHost));
    for (int row = 0; row < m; ++row)
      for (int col = 0; col < n; ++col) {
        double sum = 0;
        for (int kk = 0; kk < k; ++kk)
          sum += double(float(reference[la.index(0, row, kk)])) *
                 double(float(b[lb.index(0, kk, col)]));
        expect(float(output[ld.index(0, row, col)]) ==
                   float(T(float(alpha * sum))),
               "beta zero ignores NaN C");
      }
    // A queued, delayed host marker makes an asynchronous implementation of
    // Search reliably observable, without blocking or faulting the GPU.
    std::atomic<bool> marker{false};
    HIP(hipLaunchHostFunc(stream, delayed_marker, &marker));
    LT(hipsparseLtMatmulSearch(&handle, &plan, &alpha, compressed.data,
                               gpu_b.data, &zero, gpu_c.data, gpu_d.data,
                               nullptr, &stream, 1));
    bool completed = marker.load();
    hipError_t query = hipStreamQuery(stream);
    HIP(hipStreamSynchronize(
        stream)); // keep marker alive even if the regression fails
    expect(completed && query == hipSuccess,
           "Search must block until stream completes");
  }
  // Invalid sparsity is explicitly detectable through the public API.
  std::vector<T> bad(input.size(), T(1.0f));
  HIP(hipMemcpy(gpu_sparse.data, bad.data(), bad.size() * sizeof(T),
                hipMemcpyHostToDevice));
  LT(hipsparseLtSpMMAPruneCheck(&handle, &mm, gpu_sparse.data,
                                static_cast<int *>(invalid.data), stream));
  HIP(hipStreamSynchronize(stream));
  HIP(hipMemcpy(&valid, invalid.data, sizeof(int), hipMemcpyDeviceToHost));
  expect(valid == 1, "invalid sparsity detection");
  HIP(hipStreamDestroy(stream));
  LT(hipsparseLtMatmulPlanDestroy(&plan));
  LT(hipsparseLtMatmulAlgSelectionDestroy(&alg));
  LT(hipsparseLtMatDescriptorDestroy(&da));
  LT(hipsparseLtMatDescriptorDestroy(&db));
  LT(hipsparseLtMatDescriptorDestroy(&dc));
  LT(hipsparseLtMatDescriptorDestroy(&dd));
}
int main() {
  try {
    hipsparseLtHandle_t handle{};
    LT(hipsparseLtInit(&handle));
    int cases = 0;
    for (bool sparse_a : {false, true})
      for (auto order : {HIPSPARSE_ORDER_ROW, HIPSPARSE_ORDER_COL})
        for (bool ta : {false, true})
          for (bool tb : {false, true})
            for (int batches : {1, 2}) {
              run<float>(handle, HIP_R_32F, sparse_a, order, ta, tb, batches);
              ++cases;
              run<__half>(handle, HIP_R_16F, sparse_a, order, ta, tb, batches);
              ++cases;
              run<hip_bfloat16>(handle, HIP_R_16BF, sparse_a, order, ta, tb,
                                batches);
              ++cases;
              run<__half>(handle, HIP_R_16F, sparse_a, order, ta, tb, batches,
                          true);
              ++cases;
              run<hip_bfloat16>(handle, HIP_R_16BF, sparse_a, order, ta, tb,
                                batches, true);
              ++cases;
            }
    run<__half>(handle, HIP_R_16F, true, HIPSPARSE_ORDER_COL, true, false, 1,
                true, true);
    ++cases;
    run<hip_bfloat16>(handle, HIP_R_16BF, false, HIPSPARSE_ORDER_ROW, false,
                      true, 1, true, true);
    ++cases;
    for (int activation :
         {HIPSPARSELT_MATMUL_ACTIVATION_RELU, HIPSPARSELT_MATMUL_ACTIVATION_ABS,
          HIPSPARSELT_MATMUL_ACTIVATION_LEAKYRELU,
          HIPSPARSELT_MATMUL_ACTIVATION_GELU,
          HIPSPARSELT_MATMUL_ACTIVATION_SIGMOID,
          HIPSPARSELT_MATMUL_ACTIVATION_TANH})
      for (bool sparseA : {false, true})
        for (auto order : {HIPSPARSE_ORDER_ROW, HIPSPARSE_ORDER_COL}) {
          run<float>(handle, HIP_R_32F, sparseA, order, true, false, 2, false,
                     false, activation);
          ++cases;
          run<__half>(handle, HIP_R_16F, sparseA, order, false, true, 2, false,
                      false, activation);
          ++cases;
          run<hip_bfloat16>(handle, HIP_R_16BF, sparseA, order, true, true, 2,
                            false, false, activation);
          ++cases;
        }
    LT(hipsparseLtDestroy(&handle));
    std::printf("HIPSPARSELT_SCALAR_PASS cases=%d prune=strip,tile "
                "compressed=independently-decoded reference=dense-double\n",
                cases);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL %s\n", e.what());
    return 1;
  }
}
