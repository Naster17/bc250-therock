// SPDX-License-Identifier: MIT
// Cold hostcall setup after dynamic stream churn; printf is never executed.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call) do { auto status = (call); if (status != hipSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(status)); std::exit(1); } } while (0)

__global__ void warm(int* output, int slot) {
  if (threadIdx.x == 0) output[slot] = slot + 11;
}

__global__ void cold_hostcall(int* output, int slot, int enable_printf) {
  if (threadIdx.x == 0) {
    // A runtime predicate retains hostcall metadata without a device printf.
    if (enable_printf) printf("unused hostcall probe %d\n", slot);
    output[slot] = slot * 17 + 9;
  }
}

int main() {
  constexpr int streams_per_round = 20, rounds = 3, words = 128;
  int* device = nullptr;
  CHECK(hipMalloc(&device, words * sizeof(int)));
  for (int round = 0; round < rounds; ++round) {
    std::vector<hipStream_t> streams(streams_per_round);
    std::vector<int> input(words, 0x2555a5), output(words);
    CHECK(hipMemcpy(device, input.data(), words*sizeof(int), hipMemcpyHostToDevice));
    for (auto& stream : streams) CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    for (int i = 0; i < streams_per_round; ++i) {
      warm<<<1,32,0,streams[i]>>>(device, i+3);
      CHECK(hipGetLastError());
      CHECK(hipStreamSynchronize(streams[i]));
    }
    for (int i = 0; i < streams_per_round; ++i) {
      cold_hostcall<<<1,32,0,streams[i]>>>(device, i+3, 0);
      CHECK(hipGetLastError());
      CHECK(hipStreamSynchronize(streams[i]));
    }
    CHECK(hipMemcpy(output.data(), device, words*sizeof(int), hipMemcpyDeviceToHost));
    for (int i = 0; i < words; ++i) {
      const int expected = i >= 3 && i < streams_per_round+3 ? i*17+9 : input[i];
      if (output[i] != expected) { std::fprintf(stderr,"mismatch %d %d %d\n",i,output[i],expected); return 2; }
    }
    for (auto stream : streams) CHECK(hipStreamDestroy(stream));
  }
  CHECK(hipFree(device));
  std::puts("HOSTCALL_QUEUE_PASS rounds=3 streams=60 guarded_words=384");
}
