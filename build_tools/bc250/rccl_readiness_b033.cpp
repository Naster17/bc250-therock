// SPDX-License-Identifier: MIT
// Single-device readiness only: standard one-rank collectives are
// copies/no-ops.
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>

#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void hipcheck(hipError_t code) {
  if (code != hipSuccess)
    throw std::runtime_error(hipGetErrorString(code));
}
static void ncclcheck(ncclResult_t code) {
  if (code != ncclSuccess)
    throw std::runtime_error(ncclGetErrorString(code));
}
static void require(bool condition, const char *text) {
  if (!condition)
    throw std::runtime_error(text);
}
static void passed(const std::string &name) {
  std::cout << "PASS " << name << std::endl;
}

template <class T>
static void premul(ncclComm_t comm, hipStream_t stream, ncclDataType_t dtype,
                   const char *name) {
  for (int n : {17, 257, 1025}) {
    for (bool inplace : {false, true}) {
      for (bool device_scalar : {false, true}) {
        std::vector<T> input(n + 16, T(-23)), reference(n + 16, T(-23));
        for (int i = 0; i < n; ++i) {
          input[i + 8] = T((i % 11) - 5);
          reference[i + 8] = T(2 * ((i % 11) - 5));
        }
        T *src = nullptr, *dst = nullptr, *scalar = nullptr;
        hipcheck(hipMalloc(reinterpret_cast<void **>(&src),
                           input.size() * sizeof(T)));
        hipcheck(hipMalloc(reinterpret_cast<void **>(&dst),
                           input.size() * sizeof(T)));
        hipcheck(hipMalloc(reinterpret_cast<void **>(&scalar), sizeof(T)));
        std::vector<T> blank(n + 16, T(-23));
        T factor = T(2);
        hipcheck(hipMemcpyAsync(src, input.data(), input.size() * sizeof(T),
                                hipMemcpyHostToDevice, stream));
        hipcheck(hipMemcpyAsync(dst, blank.data(), blank.size() * sizeof(T),
                                hipMemcpyHostToDevice, stream));
        hipcheck(hipMemcpyAsync(scalar, &factor, sizeof(T),
                                hipMemcpyHostToDevice, stream));
        ncclRedOp_t op;
        ncclcheck(ncclRedOpCreatePreMulSum(
            &op, device_scalar ? scalar : &factor, dtype,
            device_scalar ? ncclScalarDevice : ncclScalarHostImmediate, comm));
        T *target = inplace ? src : dst;
        ncclcheck(
            ncclAllReduce(src + 8, target + 8, n, dtype, op, comm, stream));
        // Queue a readback on the same non-default stream; no device-wide fence
        // first.
        std::vector<T> actual(n + 16), original(n + 16);
        hipcheck(hipMemcpyAsync(actual.data(), target,
                                actual.size() * sizeof(T),
                                hipMemcpyDeviceToHost, stream));
        hipcheck(hipMemcpyAsync(original.data(), src,
                                original.size() * sizeof(T),
                                hipMemcpyDeviceToHost, stream));
        hipcheck(hipStreamSynchronize(stream));
        for (size_t i = 0; i < actual.size(); ++i) {
          require(static_cast<double>(actual[i]) ==
                      static_cast<double>(reference[i]),
                  "PreMulSum output/guard mismatch");
          require(static_cast<double>(original[i]) ==
                      static_cast<double>((inplace ? reference : input)[i]),
                  "input/guard mismatch");
        }
        ncclcheck(ncclRedOpDestroy(op, comm));
        hipcheck(hipFree(scalar));
        hipcheck(hipFree(dst));
        hipcheck(hipFree(src));
        passed(std::string("premul-") + name + "-" + std::to_string(n) +
               (inplace ? "-in" : "-out") +
               (device_scalar ? "-device-scalar" : "-host-scalar"));
      }
    }
  }
}

int main() {
  try {
    int count = 0, version = 0;
    hipcheck(hipGetDeviceCount(&count));
    require(count == 1, "one GPU required");
    hipcheck(hipSetDevice(0));
    hipDeviceProp_t props{};
    hipcheck(hipGetDeviceProperties(&props, 0));
    require(std::string(props.gcnArchName).rfind("gfx1013:", 0) == 0,
            "native gfx1013 required");
    ncclcheck(ncclGetVersion(&version));
    require(version == NCCL_VERSION_CODE, "header/library version mismatch");
    Dl_info library{};
    require(dladdr(reinterpret_cast<void *>(ncclGetVersion), &library) != 0,
            "library identity lookup failed");
    std::cout << "LIBRARY " << library.dli_fname << std::endl;
    passed("native-device-version");
    ncclUniqueId id;
    ncclcheck(ncclGetUniqueId(&id));
    ncclComm_t comm;
    ncclcheck(ncclCommInitRank(&comm, 1, id, 0));
    int ranks = 0, rank = -1, device = -1;
    ncclcheck(ncclCommCount(comm, &ranks));
    ncclcheck(ncclCommUserRank(comm, &rank));
    ncclcheck(ncclCommCuDevice(comm, &device));
    require(ranks == 1 && rank == 0 && device == 0,
            "communicator metadata mismatch");
    passed("single-rank-init-metadata");
    hipStream_t stream;
    hipcheck(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    premul<float>(comm, stream, ncclFloat32, "f32");
    premul<double>(comm, stream, ncclFloat64, "f64");
    premul<__half>(comm, stream, ncclFloat16, "f16");
    premul<hip_bfloat16>(comm, stream, ncclBfloat16, "bf16");
    premul<int32_t>(comm, stream, ncclInt32, "i32");
    premul<int64_t>(comm, stream, ncclInt64, "i64");
    // Standard world-size-one API plumbing: these are identity transfers.
    std::vector<float> input(37), actual(37, -1);
    for (size_t i = 0; i < input.size(); ++i)
      input[i] = float(i) / 4;
    float *src, *dst;
    hipcheck(hipMalloc(&src, input.size() * sizeof(float)));
    hipcheck(hipMalloc(&dst, input.size() * sizeof(float)));
    hipcheck(hipMemcpyAsync(src, input.data(), input.size() * sizeof(float),
                            hipMemcpyHostToDevice, stream));
    for (int op = 0; op < 6; ++op) {
      hipcheck(hipMemsetAsync(dst, 0xff, input.size() * sizeof(float), stream));
      if (op == 0)
        ncclcheck(ncclAllReduce(src, dst, input.size(), ncclFloat32, ncclSum,
                                comm, stream));
      if (op == 1)
        ncclcheck(ncclBroadcast(src, dst, input.size(), ncclFloat32, 0, comm,
                                stream));
      if (op == 2)
        ncclcheck(ncclReduce(src, dst, input.size(), ncclFloat32, ncclSum, 0,
                             comm, stream));
      if (op == 3)
        ncclcheck(
            ncclAllGather(src, dst, input.size(), ncclFloat32, comm, stream));
      if (op == 4)
        ncclcheck(ncclReduceScatter(src, dst, input.size(), ncclFloat32,
                                    ncclSum, comm, stream));
      if (op == 5) {
        ncclcheck(ncclGroupStart());
        ncclcheck(ncclAllReduce(src, dst, input.size(), ncclFloat32, ncclSum,
                                comm, stream));
        ncclcheck(ncclGroupEnd());
      }
      hipcheck(hipMemcpyAsync(actual.data(), dst, input.size() * sizeof(float),
                              hipMemcpyDeviceToHost, stream));
      hipcheck(hipStreamSynchronize(stream));
      require(actual == input, "single-rank identity mismatch");
      passed("single-rank-identity-" + std::to_string(op));
    }
    hipcheck(hipFree(dst));
    hipcheck(hipFree(src));
    ncclResult_t asynchronous;
    ncclcheck(ncclCommGetAsyncError(comm, &asynchronous));
    require(asynchronous == ncclSuccess, "async communicator error");
    passed("async-status");
    ncclcheck(ncclCommFinalize(comm));
    ncclcheck(ncclCommDestroy(comm));
    hipcheck(hipStreamDestroy(stream));
    passed("finalize-destroy");
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
