// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <rpp/rpp.h>
#include <rpp/rppt_tensor_audio_augmentations.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

static void hip_check(hipError_t status) {
  if (status != hipSuccess)
    throw std::runtime_error(hipGetErrorString(status));
}

static void rpp_check(RppStatus status) {
  if (status != RPP_SUCCESS)
    throw std::runtime_error("RPP audio operation failed");
}

static void compare(float actual, double expected, const char *operation,
                    size_t index) {
  if (!std::isfinite(actual) ||
      std::abs(actual - expected) > 2e-4 * std::max(1.0, std::abs(expected))) {
    fprintf(stderr, "%s mismatch at %zu: actual=%.9g expected=%.17g\n",
            operation, index, actual, expected);
    throw std::runtime_error(
        "RPP audio differs from independent CPU reference");
  }
}

static void run(RppBackend backend) {
  constexpr unsigned batches = 3, samples = 64, nfft = 16, frames = 4,
                     bins = nfft / 2 + 1;
  constexpr float sentinel = -17.0f;
  RpptDesc source(batches, 1, 1, samples, RpptDataType::F32, RpptLayout::NCHW);
  source.numDims = 2;
  std::vector<float> input(batches * samples + 16, 0.0f);
  for (unsigned n = 0; n < batches; ++n)
    for (unsigned i = 0; i < samples; ++i)
      input[n * samples + i] = float(int((i * 7 + n * 11) % 23) - 11) / 16.0f;
  float *device_input, *device_output, *coefficients, *window;
  int *lengths;
  hip_check(hipMalloc(&device_input, input.size() * sizeof(float)));
  hip_check(hipMalloc(&device_output, input.size() * sizeof(float)));
  hip_check(hipHostMalloc(&coefficients, batches * sizeof(float)));
  hip_check(hipHostMalloc(&lengths, batches * sizeof(int)));
  hip_check(hipHostMalloc(&window, nfft * sizeof(float)));
  hip_check(hipMemcpy(device_input, input.data(), input.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  coefficients[0] = 0.0f;
  coefficients[1] = 0.5f;
  coefficients[2] = 0.97f;
  lengths[0] = 16;
  lengths[1] = 24;
  lengths[2] = 31;
  std::fill_n(window, nfft, 1.0f);
  rppHandle_t handle;
  if (rppCreate(&handle, batches, 1, nullptr, backend) != rppStatusSuccess)
    throw std::runtime_error("rppCreate failed");
  const bool gpu = backend == RppBackend::RPP_HIP_BACKEND;
  auto *src = gpu ? device_input : input.data();
  std::vector<float> output(input.size(), sentinel);
  for (auto border : {RpptAudioBorderType::ZERO, RpptAudioBorderType::CLAMP,
                      RpptAudioBorderType::REFLECT}) {
    std::fill(output.begin(), output.end(), sentinel);
    hip_check(hipMemcpy(device_output, output.data(),
                        output.size() * sizeof(float), hipMemcpyHostToDevice));
    rpp_check(rppt_pre_emphasis_filter(
        src, &source, gpu ? device_output : output.data(), &source, lengths,
        coefficients, border, handle, backend));
    if (gpu)
      hip_check(hipMemcpy(output.data(), device_output,
                          output.size() * sizeof(float),
                          hipMemcpyDeviceToHost));
    for (unsigned n = 0; n < batches; ++n) {
      for (int i = 0; i < lengths[n]; ++i) {
        size_t index = n * samples + i;
        float previous = i ? input[index - 1]
                         : border == RpptAudioBorderType::CLAMP ? input[index]
                         : border == RpptAudioBorderType::REFLECT
                             ? input[index + 1]
                             : 0.0f;
        compare(output[index],
                double(input[index]) - coefficients[n] * previous,
                "pre_emphasis", index);
      }
      // HIP vector stores may write the padded tail of the final eight samples.
      for (unsigned i = (lengths[n] + 7) / 8 * 8; i < samples; ++i)
        if (output[n * samples + i] != sentinel)
          throw std::runtime_error("pre_emphasis overwrote batch padding");
    }
    printf("RPP_AUDIO_PREEMPHASIS_PASS gpu=%d border=%d\n", gpu, int(border));
  }
  // Magnitude padding is deliberately larger than every logical element.
  RpptImagePatch *dimensions;
  hip_check(hipHostMalloc(&dimensions, batches * sizeof(RpptImagePatch)));
  for (bool matrix : {false, true}) {
    auto descriptor = source;
    descriptor.numDims = matrix ? 3 : 2;
    descriptor.h = matrix ? 3 : samples;
    descriptor.w = matrix ? 7 : 1;
    descriptor.strides.hStride = matrix ? 16 : 1;
    std::vector<float> magnitudes(input.size(), 100.0f);
    for (unsigned n = 0; n < batches; ++n) {
      dimensions[n].height = matrix ? 2 + n % 2 : 37 + n * 12;
      dimensions[n].width = matrix ? 5 + n : 1;
      for (unsigned y = 0; y < dimensions[n].height; ++y)
        for (unsigned x = 0; x < dimensions[n].width; ++x) {
          size_t i = n * samples + y * descriptor.strides.hStride + x;
          magnitudes[i] =
              n == 0 ? 0.0f
                     : std::pow(10.0f, -float((y * 7 + x * 3 + n) % 19) / 2);
        }
    }
    hip_check(hipMemcpy(device_input, magnitudes.data(),
                        input.size() * sizeof(float), hipMemcpyHostToDevice));
    for (float multiplier : {10.0f, 20.0f})
      for (float reference : {0.0f, 2.0f}) {
        std::fill(output.begin(), output.end(), sentinel);
        hip_check(hipMemcpy(device_output, output.data(),
                            output.size() * sizeof(float),
                            hipMemcpyHostToDevice));
        rpp_check(rppt_to_decibels(
            gpu ? device_input : magnitudes.data(), &descriptor,
            gpu ? device_output : output.data(), &descriptor, dimensions,
            -80.0f, multiplier, reference, handle, backend));
        if (gpu)
          hip_check(hipMemcpy(output.data(), device_output,
                              output.size() * sizeof(float),
                              hipMemcpyDeviceToHost));
        for (unsigned n = 0; n < batches; ++n) {
          double ref = reference;
          if (ref == 0) {
            for (unsigned y = 0; y < dimensions[n].height; ++y)
              for (unsigned x = 0; x < dimensions[n].width; ++x)
                ref = std::max(
                    ref,
                    double(magnitudes[n * samples +
                                      y * descriptor.strides.hStride + x]));
            if (ref == 0)
              ref = 1; // Public CPU backend's all-zero reference policy.
          }
          std::vector<bool> written(samples, false);
          for (unsigned y = 0; y < dimensions[n].height; ++y)
            for (unsigned x = 0; x < dimensions[n].width; ++x) {
              auto offset = y * descriptor.strides.hStride + x;
              auto i = n * samples + offset;
              written[offset] = true;
              double expected =
                  multiplier *
                  std::log10(std::max(double(magnitudes[i]) / ref,
                                      std::pow(10.0, -80.0 / multiplier)));
              if (!std::isfinite(output[i]) ||
                  std::abs(output[i] - expected) > 3e-4) {
                fprintf(stderr,
                        "decibels gpu=%d matrix=%d n=%u i=%zu got=%g "
                        "expected=%g ref=%g\n",
                        gpu, matrix, n, i, output[i], expected, ref);
                throw std::runtime_error(
                    "decibels differs from independent CPU reference");
              }
            }
          if (gpu && !matrix)
            for (unsigned i = dimensions[n].height;
                 i < (dimensions[n].height + 7) / 8 * 8; ++i)
              written[i] = true;
          for (unsigned i = 0; i < samples; ++i)
            if (!written[i] && output[n * samples + i] != sentinel)
              throw std::runtime_error("decibels overwrote row/batch padding");
        }
        for (size_t i = batches * samples; i < output.size(); ++i)
          if (output[i] != sentinel)
            throw std::runtime_error("decibels overwrote allocation guard");
        printf("RPP_AUDIO_DECIBELS_PASS gpu=%d matrix=%d multiplier=%g "
               "reference=%g\n",
               gpu, matrix, multiplier, reference);
      }
  }
  hip_check(hipHostFree(dimensions));
  hip_check(hipMemcpy(device_input, input.data(), input.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  for (unsigned n = 0; n < batches; ++n)
    lengths[n] = samples;
  for (auto layout : {RpptLayout::NTF, RpptLayout::NFT}) {
    RpptDesc destination(batches, 1, layout == RpptLayout::NTF ? frames : bins,
                         layout == RpptLayout::NTF ? bins : frames,
                         RpptDataType::F32, RpptLayout::NCHW);
    destination.layout = layout;
    destination.numDims = 3;
    destination.strides.nStride = frames * bins + 16;
    std::fill(output.begin(), output.end(), sentinel);
    hip_check(hipMemcpy(device_output, output.data(),
                        output.size() * sizeof(float), hipMemcpyHostToDevice));
    rpp_check(rppt_spectrogram(
        src, &source, gpu ? device_output : output.data(), &destination,
        lengths, false, false, window, nfft, 2, nfft, nfft, handle, backend));
    if (gpu)
      hip_check(hipMemcpy(output.data(), device_output,
                          output.size() * sizeof(float),
                          hipMemcpyDeviceToHost));
    for (unsigned n = 0; n < batches; ++n) {
      for (unsigned frame = 0; frame < frames; ++frame)
        for (unsigned frequency = 0; frequency < bins; ++frequency) {
          double real = 0.0, imaginary = 0.0;
          for (unsigned i = 0; i < nfft; ++i) {
            double angle = -2.0 * std::acos(-1.0) * frequency * i / nfft;
            double sample = input[n * samples + frame * nfft + i];
            real += sample * std::cos(angle);
            imaginary += sample * std::sin(angle);
          }
          size_t index =
              n * destination.strides.nStride +
              (layout == RpptLayout::NTF ? frame * bins + frequency
                                         : frequency * frames + frame);
          compare(output[index], real * real + imaginary * imaginary,
                  "spectrogram", index);
        }
      for (unsigned i = frames * bins; i < destination.strides.nStride; ++i)
        if (output[n * destination.strides.nStride + i] != sentinel)
          throw std::runtime_error("spectrogram overwrote batch padding");
    }
    for (size_t i = batches * destination.strides.nStride; i < output.size();
         ++i)
      if (output[i] != sentinel)
        throw std::runtime_error("spectrogram overwrote allocation guard");
    printf("RPP_AUDIO_SPECTROGRAM_PASS gpu=%d layout=%s\n", gpu,
           layout == RpptLayout::NTF ? "NTF" : "NFT");
  }
  if (rppDestroy(handle, backend) != rppStatusSuccess)
    throw std::runtime_error("rppDestroy failed");
  hip_check(hipHostFree(window));
  hip_check(hipHostFree(lengths));
  hip_check(hipHostFree(coefficients));
  hip_check(hipFree(device_output));
  hip_check(hipFree(device_input));
}

int main() {
  try {
    run(RppBackend::RPP_HOST_BACKEND);
    run(RppBackend::RPP_HIP_BACKEND);
    puts("RPP_AUDIO_NATIVE_GFX1013_PASS");
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
  return 0;
}
