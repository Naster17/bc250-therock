// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <rpp/rpp.h>
#include <rpp/rppt_tensor_color_augmentations.h>
#include <rpp/rppt_tensor_geometric_augmentations.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

static void hc(hipError_t s) {
  if (s != hipSuccess)
    throw std::runtime_error(hipGetErrorString(s));
}
static void rc(RppStatus s) {
  if (s != RPP_SUCCESS)
    throw std::runtime_error("RPP operation failed");
}
static RpptDesc desc(RpptLayout layout) {
  RpptDesc d(2, 3, 9, 13, RpptDataType::F32, layout);
  d.strides.wStride = layout == RpptLayout::NHWC ? 3 : 1;
  d.strides.hStride = 16 * d.strides.wStride;
  d.strides.cStride = layout == RpptLayout::NHWC ? 1 : 9 * 16;
  d.strides.nStride = 9 * 16 * 3;
  return d;
}
static size_t index(const RpptDesc &d, int n, int c, int y, int x) {
  return n * d.strides.nStride + c * d.strides.cStride + y * d.strides.hStride +
         x * d.strides.wStride;
}
static void run(RpptLayout sl, RpptLayout dl) {
  auto s = desc(sl), d = desc(dl);
  const size_t count = 2 * s.strides.nStride + 32;
  std::vector<float> a(count, 0.25f), b(count, 0.5f), out(count, -17);
  for (int n = 0; n < 2; ++n)
    for (int c = 0; c < 3; ++c)
      for (int y = 0; y < 9; ++y)
        for (int x = 0; x < 13; ++x) {
          size_t i = index(s, n, c, y, x);
          a[i] = ((n * 11 + c * 3 + y * 5 + x * 7) % 31) / 30.f;
          b[i] = 1 - a[i];
        }
  float *ga, *gb, *go, *factor, *center;
  RpptROI *roi;
  hc(hipMalloc(&ga, count * sizeof(float)));
  hc(hipMalloc(&gb, count * sizeof(float)));
  hc(hipMalloc(&go, count * sizeof(float)));
  hc(hipHostMalloc(&factor, 2 * sizeof(float)));
  hc(hipHostMalloc(&center, 2 * sizeof(float)));
  hc(hipMalloc(&roi, 2 * sizeof(RpptROI)));
  RpptROI host_roi[2]{};
  for (auto &r : host_roi)
    r.xywhROI = {{0, 0}, 13, 9};
  hc(hipMemcpy(ga, a.data(), count * sizeof(float), hipMemcpyHostToDevice));
  hc(hipMemcpy(gb, b.data(), count * sizeof(float), hipMemcpyHostToDevice));
  hc(hipMemcpy(roi, host_roi, sizeof(host_roi), hipMemcpyHostToDevice));
  rppHandle_t handle;
  if (rppCreate(&handle, 2, 0, nullptr, RppBackend::RPP_HIP_BACKEND) !=
      rppStatusSuccess)
    throw std::runtime_error("rppCreate failed");
  auto clear = [&] {
    std::fill(out.begin(), out.end(), -17);
    hc(hipMemcpy(go, out.data(), count * sizeof(float), hipMemcpyHostToDevice));
  };
  auto verify = [&](const char *name, auto reference, int width = 13,
                    int height = 9) {
    hc(hipMemcpy(out.data(), go, count * sizeof(float), hipMemcpyDeviceToHost));
    for (int n = 0; n < 2; ++n)
      for (int c = 0; c < 3; ++c)
        for (int y = 0; y < height; ++y)
          for (int x = 0; x < width; ++x) {
            auto i = index(d, n, c, y, x);
            double expected = reference(n, c, y, x);
            if (!std::isfinite(out[i]) || std::abs(out[i] - expected) > 2e-6) {
              fprintf(stderr, "%s src=%d dst=%d at %zu got=%g expected=%g\n",
                      name, int(sl), int(dl), i, out[i], expected);
              throw std::runtime_error("RPP independent reference mismatch");
            }
          }
    for (size_t i = count - 32; i < count; ++i)
      if (out[i] != -17)
        throw std::runtime_error("RPP allocation guard changed");
    printf("RPP_EXTENDED_PASS %s src=%d dst=%d\n", name, int(sl), int(dl));
  };
  clear();
  factor[0] = 0.25;
  factor[1] = 0.75;
  rc(rppt_blend(ga, gb, &s, go, &d, factor, roi, RpptRoiType::XYWH, handle,
                RppBackend::RPP_HIP_BACKEND));
  verify("blend", [&](int n, int c, int y, int x) {
    auto i = index(s, n, c, y, x);
    return double(a[i]) * factor[n] + double(b[i]) * (1 - factor[n]);
  });
  clear();
  factor[0] = 0.5;
  factor[1] = 1.5;
  center[0] = 63.75;
  center[1] = 127.5;
  rc(rppt_contrast(ga, &s, go, &d, factor, center, roi, RpptRoiType::XYWH,
                   handle, RppBackend::RPP_HIP_BACKEND));
  verify("contrast", [&](int n, int c, int y, int x) {
    return std::clamp((double(a[index(s, n, c, y, x)]) - center[n] / 255.) *
                              factor[n] +
                          center[n] / 255.,
                      0., 1.);
  });
  clear();
  factor[0] = 0.0;
  factor[1] = 1.0;
  rc(rppt_exposure(ga, &s, go, &d, factor, roi, RpptRoiType::XYWH, handle,
                   RppBackend::RPP_HIP_BACKEND));
  verify("exposure", [&](int n, int c, int y, int x) {
    return std::min(double(a[index(s, n, c, y, x)]) * std::exp2(factor[n]), 1.);
  });
  clear();
  for (auto &r : host_roi)
    r.xywhROI = {{2, 1}, 7, 5};
  hc(hipMemcpy(roi, host_roi, sizeof(host_roi), hipMemcpyHostToDevice));
  rc(rppt_crop(ga, &s, go, &d, roi, RpptRoiType::XYWH, handle,
               RppBackend::RPP_HIP_BACKEND));
  verify(
      "offset_crop",
      [&](int n, int c, int y, int x) {
        return double(a[index(s, n, c, y + 1, x + 2)]);
      },
      7, 5);
  if (rppDestroy(handle, RppBackend::RPP_HIP_BACKEND) != rppStatusSuccess)
    throw std::runtime_error("rppDestroy failed");
  hc(hipFree(roi));
  hc(hipHostFree(center));
  hc(hipHostFree(factor));
  hc(hipFree(go));
  hc(hipFree(gb));
  hc(hipFree(ga));
}
int main() {
  try {
    for (auto s : {RpptLayout::NHWC, RpptLayout::NCHW})
      for (auto d : {RpptLayout::NHWC, RpptLayout::NCHW})
        run(s, d);
    puts("RPP_EXTENDED_NATIVE_GFX1013_PASS 16 cases");
    return 0;
  } catch (const std::exception &e) {
    fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
