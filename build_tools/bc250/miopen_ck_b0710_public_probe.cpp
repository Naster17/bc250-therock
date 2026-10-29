// SPDX-License-Identifier: MIT
#include "primitives_n216_common.hpp"
#include <array>
using Value = _Float16;
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <miopen/conv/problem_description.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/miopen.h>
#include <miopen/solver/ck_impl_interface.hpp>
#include <miopen/solver_id.hpp>
void mi(miopenStatus_t s) {
  require(s == miopenStatusSuccess, miopenGetErrorString(s));
}
struct Tensor {
  miopenTensorDescriptor_t d{};
  Tensor(int n, int c, int h, int w, bool nhwc) {
    mi(miopenCreateTensorDescriptor(&d));
    int lengths[]{n, c, h, w};
    mi(miopenSetNdTensorDescriptorWithLayout(
        d, miopenHalf, nhwc ? miopenTensorNHWC : miopenTensorNCHW, lengths, 4));
  }
  ~Tensor() { miopenDestroyTensorDescriptor(d); }
};
int main(int argc, char **) {
  try {
    const bool negative = argc > 1;
    hipDeviceProp_t props{};
    check(hipGetDeviceProperties(&props, 0));
    require(std::string(props.gcnArchName) == "gfx1013:xnack-",
            "native architecture");
    miopen::solver::Id solver("ConvHipImplicitGemmGroupFwdXdlops");
    require(solver.IsValid(), "solver identity");
    miopenHandle_t handle;
    mi(miopenCreate(&handle));
    Stream stream;
    size_t executions = 0, refusals = 0;
    double max_error = 0;
    for (bool nhwc : {false, true})
      for (int g : {1, 3})
        for (int f : {1, 3})
          for (bool custom : {false, true})
            for (bool dyadic : {false, true}) {
              const int n = 2, c = 4, k = 4, h = 5, w = 7, pad = f / 2;
              Tensor xd(n, g * c, h, w, nhwc), wd(g * k, c, f, f, nhwc),
                  yd(n, g * k, h, w, nhwc);
              miopenConvolutionDescriptor_t conv;
              mi(miopenCreateConvolutionDescriptor(&conv));
              mi(miopenInitConvolutionDescriptor(conv, miopenConvolution, pad,
                                                 pad, 1, 1, 1, 1));
              mi(miopenSetConvolutionGroupCount(conv, g));
              auto st = custom ? stream.value : hipStream_t{};
              mi(miopenSetStream(handle, st));
              size_t workspace = 0;
              auto status = miopenConvolutionForwardGetSolutionWorkspaceSize(
                  handle, wd.d, xd.d, conv, yd.d, solver.Value(), &workspace);
              if (negative) {
                status = miopenConvolutionForwardCompileSolution(
                    handle, wd.d, xd.d, conv, yd.d, solver.Value());
                require(status != miopenStatusSuccess,
                        "private solver admitted without plugin");
                ++refusals;
                miopenDestroyConvolutionDescriptor(conv);
                continue;
              }
              mi(status);
              require(workspace < 64 * 1024 * 1024, "workspace bound");
              auto at = [=](int b, int ch, int y, int x, int channels,
                            int height, int width) {
                return nhwc ? ((b * height + y) * width + x) * channels + ch
                            : ((b * channels + ch) * height + y) * width + x;
              };
              std::vector<Value> a(n * g * c * h * w), b(g * k * c * f * f);
              for (size_t i = 0; i < a.size(); ++i)
                a[i] =
                    float(int((i * 13 + 3) % 19) - 9) / (dyadic ? 8.f : 11.f);
              for (size_t i = 0; i < b.size(); ++i)
                b[i] = float(int((i * 7 + 5) % 17) - 8) / (dyadic ? 8.f : 11.f);
              std::vector<double> gold(n * g * k * h * w);
              for (int batch = 0; batch < n; ++batch)
                for (int group = 0; group < g; ++group)
                  for (int out = 0; out < k; ++out)
                    for (int y = 0; y < h; ++y)
                      for (int x = 0; x < w; ++x) {
                        double sum = 0;
                        for (int ch = 0; ch < c; ++ch)
                          for (int fy = 0; fy < f; ++fy)
                            for (int fx = 0; fx < f; ++fx) {
                              int iy = y + fy - pad, ix = x + fx - pad;
                              if (iy >= 0 && iy < h && ix >= 0 && ix < w)
                                sum += double(a[at(batch, group * c + ch, iy,
                                                   ix, g * c, h, w)]) *
                                       double(b[at(group * k + out, ch, fy, fx,
                                                   c, f, f)]);
                            }
                        gold[at(batch, group * k + out, y, x, g * k, h, w)] =
                            sum;
                      }
              Buffer<Value> da(a, st), db(b, st), de(gold.size());
              Buffer<unsigned char> work(workspace);
              mi(miopenConvolutionForwardImmediate(
                  handle, wd.d, db.data, xd.d, da.data, conv, yd.d, de.data,
                  workspace ? work.data : nullptr, workspace, solver.Value()));
              auto actual = de.read(gold.size(), st);
              for (size_t i = 0; i < gold.size(); ++i) {
                double error = std::abs(double(actual[i]) - gold[i]);
                double rounded = double(Value(float(gold[i])));
                double spacing = std::max(
                    std::ldexp(1., -24),
                    std::ldexp(1., gold[i] == 0
                                       ? -24
                                       : std::ilogb(std::abs(gold[i])) - 10));
                require(std::isfinite(double(actual[i])) &&
                            (dyadic ? double(actual[i]) == rounded
                                    : error <= .5 * spacing + 2e-5 +
                                                   5e-5 * std::abs(gold[i])),
                        "public solver FP16 CPU reference mismatch");
                max_error = std::max(max_error, error);
              }
              work.read(workspace, st);
              da.unchanged(a, st);
              db.unchanged(b, st);
              ++executions;
              mi(miopenDestroyConvolutionDescriptor(conv));
            }
    mi(miopenDestroy(handle));
    std::cout << "{\"status\":\"passed\",\"executions\":" << executions
              << ",\"refusals\":" << refusals
              << ",\"max_abs_error\":" << max_error << ",\"guards\":true}\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
