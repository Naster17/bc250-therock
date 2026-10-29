// SPDX-License-Identifier: MIT
#include "primitives_n216_common.hpp"
#include <array>
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
        d, miopenFloat, nhwc ? miopenTensorNHWC : miopenTensorNCHW, lengths,
        4));
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
              std::vector<float> a(n * g * c * h * w), b(g * k * c * f * f);
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
              Buffer<float> da(a, st), db(b, st), de(gold.size());
              Buffer<unsigned char> work(workspace);
              mi(miopenConvolutionForwardImmediate(
                  handle, wd.d, db.data, xd.d, da.data, conv, yd.d, de.data,
                  workspace ? work.data : nullptr, workspace, solver.Value()));
              auto actual = de.read(gold.size(), st);
              for (size_t i = 0; i < gold.size(); ++i) {
                double error = std::abs(double(actual[i]) - gold[i]);
                require(std::isfinite(actual[i]) &&
                            error <=
                                (dyadic ? 0. : 2e-5 + 5e-5 * std::abs(gold[i])),
                        "forced solver CPU reference mismatch");
                max_error = std::max(max_error, error);
              }
              work.read(workspace, st);
              da.unchanged(a, st);
              db.unchanged(b, st);
              // Exercise the facade using real matching problem objects.
              miopen::conv::ProblemDescription problem(
                  miopen::deref(xd.d), miopen::deref(wd.d), miopen::deref(yd.d),
                  miopen::deref(conv), miopen::conv::Direction::Forward);
              const char *folder = std::getenv("MIOPEN_CK_LIB_PATH");
              require(folder != nullptr, "private path");
              std::string path =
                  std::string(folder) + "/libMIOpenCKGroupedConv_gfx1013.so";
              void *plugin = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
              require(plugin != nullptr, "plugin dlopen");
              auto applicable =
                  reinterpret_cast<decltype(&ck_impl_fwd_is_applicable)>(
                      dlsym(plugin, "ck_impl_fwd_is_applicable"));
              bool allowed = true;
              require(applicable(&problem, miopenHalf, false, &allowed) ==
                              CK_IMPL_STATUS_SUCCESS &&
                          !allowed,
                      "FP16 refusal");
              allowed = true;
              require(applicable(&problem, miopenFloat, true, &allowed) ==
                              CK_IMPL_STATUS_SUCCESS &&
                          !allowed,
                      "TF32 refusal");
              require(applicable(nullptr, miopenFloat, false, &allowed) ==
                          CK_IMPL_STATUS_BAD_PARAM,
                      "null problem refusal");
              if (executions == 0) {
                miopen::ExecutionContext context(&miopen::deref(handle));
                auto free_list =
                    reinterpret_cast<decltype(&ck_impl_kernel_list_free)>(
                        dlsym(plugin, "ck_impl_kernel_list_free"));
                auto size_list =
                    reinterpret_cast<decltype(&ck_impl_kernel_list_size)>(
                        dlsym(plugin, "ck_impl_kernel_list_size"));
                for (const char *prefix :
                     {"bwd", "wrw", "3d_fwd", "3d_bwd", "3d_wrw",
                      "fused_bias_activ", "fused_bias_res_add_activ",
                      "fused_grp_activ", "fused_grp_bias_activ",
                      "depthwise_fwd", "depthwise_bwd_data"}) {
                  std::string base = std::string("ck_impl_") + prefix + "_";
                  auto app =
                      reinterpret_cast<decltype(&ck_impl_fwd_is_applicable)>(
                          dlsym(plugin, (base + "is_applicable").c_str()));
                  auto args = reinterpret_cast<
                      decltype(&ck_impl_fwd_is_args_supported)>(
                      dlsym(plugin, (base + "is_args_supported").c_str()));
                  auto fill = reinterpret_cast<
                      decltype(&ck_impl_fwd_fill_valid_kernels)>(
                      dlsym(plugin, (base + "fill_valid_kernels").c_str()));
                  auto space = reinterpret_cast<
                      decltype(&ck_impl_fwd_get_workspace_size)>(
                      dlsym(plugin, (base + "get_workspace_size").c_str()));
                  auto solution =
                      reinterpret_cast<decltype(&ck_impl_fwd_get_solution)>(
                          dlsym(plugin, (base + "get_solution").c_str()));
                  require(app && args && fill && space && solution,
                          "required solver ABI");
                  allowed = true;
                  require(app(&problem, miopenFloat, false, &allowed) ==
                                  CK_IMPL_STATUS_SUCCESS &&
                              !allowed,
                          "unsupported group applicability");
                  allowed = true;
                  require(args(&problem, "invalid", miopenFloat, false,
                               &allowed) == CK_IMPL_STATUS_SUCCESS &&
                              !allowed,
                          "unsupported group arguments");
                  CKKernelListHandle *list = nullptr;
                  size_t count = 1;
                  require(fill(&problem, miopenFloat, false, &list) ==
                                  CK_IMPL_STATUS_SUCCESS &&
                              list,
                          "unsupported group list");
                  require(size_list(list, &count) == CK_IMPL_STATUS_SUCCESS &&
                              count == 0,
                          "unsupported empty list");
                  free_list(list);
                  size_t bytes = 123;
                  require(space(&problem, miopenFloat, false, &bytes) ==
                                  CK_IMPL_STATUS_INVALID_VALUE &&
                              bytes == 123,
                          "unsupported workspace untouched");
                  miopen::solver::ConvSolution *result = nullptr;
                  require(solution(&context, &problem, "invalid", false,
                                   &result) == CK_IMPL_STATUS_INVALID_VALUE &&
                              !result,
                          "unsupported solution untouched");
                  refusals += 5;
                }
                mi(miopenSetTensorCastType(xd.d, miopenFloat8_fnuz));
                miopen::conv::ProblemDescription casted(
                    miopen::deref(xd.d), miopen::deref(wd.d),
                    miopen::deref(yd.d), miopen::deref(conv),
                    miopen::conv::Direction::Forward);
                allowed = true;
                require(applicable(&casted, miopenFloat, false, &allowed) ==
                                CK_IMPL_STATUS_SUCCESS &&
                            !allowed,
                        "casted tensor refusal");
                ++refusals;
              }
              dlclose(plugin);
              refusals += 3;
              ++executions;
              miopenDestroyConvolutionDescriptor(conv);
            }
    mi(miopenDestroy(handle));
    std::cout << "{\"status\":\"passed\",\"solver_id\":" << solver.Value()
              << ",\"executions\":" << executions
              << ",\"refusals\":" << refusals
              << ",\"max_abs_error\":" << max_error << ",\"guards\":true}\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
