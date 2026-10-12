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

static void hip_check(hipError_t code) {
  if (code != hipSuccess)
    throw std::runtime_error(hipGetErrorString(code));
}

static void run(RpptLayout layout) {
  constexpr unsigned batches = 2, channels = 3, height = 9, width = 13;
  constexpr unsigned padded_width = 16;
  RpptDesc descriptor(batches, channels, height, width, RpptDataType::F32,
                      layout);
  descriptor.strides.wStride = layout == RpptLayout::NHWC ? channels : 1;
  descriptor.strides.hStride = padded_width * descriptor.strides.wStride;
  descriptor.strides.cStride =
      layout == RpptLayout::NHWC ? 1 : height * descriptor.strides.hStride;
  descriptor.strides.nStride = height * padded_width * channels;
  const size_t count = batches * descriptor.strides.nStride;
  std::vector<float> input(count + 16, 0.25f), result(count + 16, -17.0f);
  auto index = [&](unsigned n, unsigned c, unsigned y, unsigned x) {
    return n * descriptor.strides.nStride + c * descriptor.strides.cStride +
           y * descriptor.strides.hStride + x * descriptor.strides.wStride;
  };
  for (unsigned n = 0; n < batches; ++n)
    for (unsigned c = 0; c < channels; ++c)
      for (unsigned y = 0; y < height; ++y)
        for (unsigned x = 0; x < width; ++x)
          input[index(n, c, y, x)] =
              (float)((n + c * 3 + y * 5 + x * 7) % 17) / 16;

  float *gpu_input, *gpu_output, *alpha, *beta;
  RpptROI *rois;
  unsigned *horizontal, *vertical;
  hip_check(hipMalloc(&gpu_input, input.size() * sizeof(float)));
  hip_check(hipMalloc(&gpu_output, result.size() * sizeof(float)));
  hip_check(hipHostMalloc(&alpha, batches * sizeof(float)));
  hip_check(hipHostMalloc(&beta, batches * sizeof(float)));
  hip_check(hipHostMalloc(&horizontal, batches * sizeof(unsigned)));
  hip_check(hipHostMalloc(&vertical, batches * sizeof(unsigned)));
  hip_check(hipMalloc(&rois, batches * sizeof(RpptROI)));
  std::vector<RpptROI> host_rois(batches);
  for (unsigned n = 0; n < batches; ++n) {
    alpha[n] = n == 0 ? 0.5f : 1.5f;
    beta[n] = 31.875f;
    horizontal[n] = n == 0;
    vertical[n] = n == 1;
    host_rois[n].xywhROI = {{0, 0}, width, height};
  }
  hip_check(hipMemcpy(gpu_input, input.data(), input.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  hip_check(hipMemcpy(gpu_output, result.data(), result.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  hip_check(hipMemcpy(rois, host_rois.data(), batches * sizeof(RpptROI),
                      hipMemcpyHostToDevice));
  rppHandle_t handle;
  if (rppCreate(&handle, batches, 0, nullptr, RppBackend::RPP_HIP_BACKEND) !=
      rppStatusSuccess)
    throw std::runtime_error("rppCreate failed");
  if (rppt_brightness(gpu_input, &descriptor, gpu_output, &descriptor, alpha,
                      beta, rois, RpptRoiType::XYWH, handle,
                      RppBackend::RPP_HIP_BACKEND) != RPP_SUCCESS)
    throw std::runtime_error("RPP brightness failed");
  hip_check(hipMemcpy(result.data(), gpu_output, result.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  for (unsigned n = 0; n < batches; ++n)
    for (unsigned c = 0; c < channels; ++c)
      for (unsigned y = 0; y < height; ++y)
        for (unsigned x = 0; x < width; ++x) {
          size_t i = index(n, c, y, x);
          float expected = std::clamp(input[i] * alpha[n] + 0.125f, 0.0f, 1.0f);
          if (std::abs(result[i] - expected) > 1e-6f)
            throw std::runtime_error("RPP brightness differs from CPU");
        }
  if (rppt_flip(gpu_input, &descriptor, gpu_output, &descriptor, horizontal,
                vertical, rois, RpptRoiType::XYWH, handle,
                RppBackend::RPP_HIP_BACKEND) != RPP_SUCCESS)
    throw std::runtime_error("RPP flip failed");
  hip_check(hipMemcpy(result.data(), gpu_output, result.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  for (unsigned n = 0; n < batches; ++n)
    for (unsigned c = 0; c < channels; ++c)
      for (unsigned y = 0; y < height; ++y)
        for (unsigned x = 0; x < width; ++x) {
          unsigned source_x = horizontal[n] ? width - 1 - x : x;
          unsigned source_y = vertical[n] ? height - 1 - y : y;
          if (result[index(n, c, y, x)] !=
              input[index(n, c, source_y, source_x)])
            throw std::runtime_error("RPP flip differs from CPU");
        }
  for (size_t i = count; i < result.size(); ++i)
    if (result[i] != -17.0f)
      throw std::runtime_error("RPP changed an allocation sentinel");
  if (rppDestroy(handle, RppBackend::RPP_HIP_BACKEND) != rppStatusSuccess)
    throw std::runtime_error("rppDestroy failed");
  hip_check(hipFree(rois));
  hip_check(hipHostFree(vertical));
  hip_check(hipHostFree(horizontal));
  hip_check(hipHostFree(beta));
  hip_check(hipHostFree(alpha));
  hip_check(hipFree(gpu_output));
  hip_check(hipFree(gpu_input));
  printf("RPP_IMAGE_PASS layout=%s\n",
         layout == RpptLayout::NHWC ? "NHWC" : "NCHW");
}

int main() {
  try {
    run(RpptLayout::NHWC);
    run(RpptLayout::NCHW);
    puts("RPP_NATIVE_GFX1013_PASS");
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
  return 0;
}
