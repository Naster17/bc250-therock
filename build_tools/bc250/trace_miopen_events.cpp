// SPDX-License-Identifier: MIT
// Test-only observer. Forward HIP event timing unchanged, retaining
// status/value.
#include <cstdio>
#include <dlfcn.h>

extern "C" int hipEventElapsedTime(float *ms, void *start, void *stop) {
  using Fn = int (*)(float *, void *, void *);
  static const auto real = [] {
    void *sym = dlsym(RTLD_NEXT, "hipEventElapsedTime");
    if (!sym) {
      static void *hip = dlopen("libamdhip64.so", RTLD_NOW | RTLD_LOCAL);
      if (hip)
        sym = dlsym(hip, "hipEventElapsedTime");
    }
    return reinterpret_cast<Fn>(sym);
  }();
  if (!real)
    return 3;
  const int status = real(ms, start, stop);
  std::fprintf(stderr, "A01_EVENT status=%d elapsed_ms=%.9g start=%p stop=%p\n",
               status, ms ? *ms : -1.0f, start, stop);
  return status;
}

#include <cstdint>
#include <cstdlib>

extern "C" int hipExtModuleLaunchKernel(void *fun, uint32_t gx, uint32_t gy,
                                        uint32_t gz, uint32_t lx, uint32_t ly,
                                        uint32_t lz, size_t shared,
                                        void *stream, void **params,
                                        void **extra, void *start, void *stop,
                                        uint32_t flags) {
  using Launch = int (*)(void *, uint32_t, uint32_t, uint32_t, uint32_t,
                         uint32_t, uint32_t, size_t, void *, void **, void **,
                         void *, void *, uint32_t);
  using Record = int (*)(void *, void *);
  static void *hip = dlopen("libamdhip64.so", RTLD_NOW | RTLD_LOCAL);
  static const auto launch =
      reinterpret_cast<Launch>(dlsym(hip, "hipExtModuleLaunchKernel"));
  static const auto record =
      reinterpret_cast<Record>(dlsym(hip, "hipEventRecord"));
  if (!launch || !record)
    return 3;
  const bool bracket = std::getenv("BC250_A01_BRACKET_EVENTS") != nullptr;
  if (!bracket)
    return launch(fun, gx, gy, gz, lx, ly, lz, shared, stream, params, extra,
                  start, stop, flags);
  int status = start ? record(start, stream) : 0;
  if (status != 0)
    return status;
  status = launch(fun, gx, gy, gz, lx, ly, lz, shared, stream, params, extra,
                  nullptr, nullptr, flags);
  if (status != 0)
    return status;
  return stop ? record(stop, stream) : 0;
}
