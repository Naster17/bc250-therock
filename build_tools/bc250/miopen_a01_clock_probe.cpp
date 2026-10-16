// SPDX-License-Identifier: MIT
// Compare shader timestamps with the kernel's KFD calibration clock.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <hip/hip_runtime.h>
#include <sys/ioctl.h>
#include <thread>
#include <unistd.h>

#define HIP(call)                                                              \
  do {                                                                         \
    auto s = call;                                                             \
    if (s != hipSuccess) {                                                     \
      std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(s));           \
      return 1;                                                                \
    }                                                                          \
  } while (0)
struct Counters {
  uint64_t gpu, cpu, system, frequency;
  uint32_t id, pad;
};
__global__ void clocks(uint64_t *out) {
  out[0] = __builtin_amdgcn_s_memrealtime();
  out[1] = __builtin_amdgcn_s_memtime();
}
__global__ void wait_ticks(uint64_t *out, uint64_t ticks) {
  const uint64_t start = __builtin_amdgcn_s_memrealtime();
  uint64_t stop;
  do {
    stop = __builtin_amdgcn_s_memrealtime();
  } while (stop - start < ticks);
  out[0] = start;
  out[1] = stop;
}

int reference() {
  int khz;
  HIP(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, 0));
  if (khz <= 0)
    return 4;
  uint64_t *device;
  HIP(hipMalloc(&device, 16));
  for (int custom = 0; custom < 2; ++custom) {
    hipStream_t stream = nullptr;
    if (custom)
      HIP(hipStreamCreate(&stream));
    wait_ticks<<<1, 1, 0, stream>>>(device, 1000);
    HIP(hipStreamSynchronize(stream));
    for (int target : {1, 2, 5}) {
      hipEvent_t start, stop;
      HIP(hipEventCreate(&start));
      HIP(hipEventCreate(&stop));
      HIP(hipEventRecord(start, stream));
      wait_ticks<<<1, 1, 0, stream>>>(device, uint64_t(khz) * target);
      HIP(hipEventRecord(stop, stream));
      HIP(hipEventSynchronize(stop));
      float initial, repeated;
      HIP(hipEventElapsedTime(&initial, start, stop));
      uint64_t host[2];
      HIP(hipMemcpy(host, device, 16, hipMemcpyDeviceToHost));
      // Cross ROCr's clock resynchronization horizon before a new query.
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      wait_ticks<<<1, 1, 0, stream>>>(device, 1000);
      HIP(hipStreamSynchronize(stream));
      HIP(hipEventElapsedTime(&repeated, start, stop));
      double raw_ms = double(host[1] - host[0]) / khz;
      std::printf("{\"custom\":%d,\"target_ms\":%d,\"raw_ms\":%.9g,\"event_"
                  "ms\":%.9g,\"repeated_ms\":%.9g,\"khz\":%d}\n",
                  custom, target, raw_ms, initial, repeated, khz);
      if (initial <= 0 || repeated <= 0 ||
          std::abs(initial - raw_ms) > 0.1 + raw_ms * 0.03 ||
          std::abs(initial - repeated) > 0.001)
        return 5;
      HIP(hipEventDestroy(start));
      HIP(hipEventDestroy(stop));
    }
    if (custom)
      HIP(hipStreamDestroy(stream));
  }
  HIP(hipFree(device));
  return 0;
}

int main(int argc, char **) {
  if (argc > 1)
    return reference();
  uint32_t id = 0;
  std::ifstream("/sys/class/kfd/kfd/topology/nodes/1/gpu_id") >> id;
  uint64_t *device;
  HIP(hipMalloc(&device, 16));
  int fd = open("/dev/kfd", O_RDWR | O_CLOEXEC);
  if (fd < 0 || id == 0)
    return 2;
  for (int i = 0; i < 8; ++i) {
    Counters before{}, after{};
    before.id = after.id = id;
    if (ioctl(fd, _IOWR('K', 0x05, Counters), &before))
      return 3;
    clocks<<<1, 1>>>(device);
    uint64_t host[2];
    HIP(hipGetLastError());
    HIP(hipMemcpy(host, device, 16, hipMemcpyDeviceToHost));
    if (ioctl(fd, _IOWR('K', 0x05, Counters), &after))
      return 3;
    std::printf(
        "{\"before\":%llu,\"realtime\":%llu,\"memtime\":%llu,\"after\":%llu,"
        "\"system_before\":%llu,\"system_after\":%llu,\"frequency\":%llu}\n",
        (unsigned long long)before.gpu, (unsigned long long)host[0],
        (unsigned long long)host[1], (unsigned long long)after.gpu,
        (unsigned long long)before.system, (unsigned long long)after.system,
        (unsigned long long)after.frequency);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  HIP(hipFree(device));
  close(fd);
}
