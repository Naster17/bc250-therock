// SPDX-License-Identifier: MIT
// Explicit per-ID invokers, scalar admission and pre-dispatch runtime
// contracts.
#include "primitives_n216_common.hpp"
#include <array>
#include <chrono>
#include <dlfcn.h>
#include <functional>
#include <iostream>
#include <map>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/conv_solution.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/handle.hpp>
#include <miopen/solver/ck_impl_interface.hpp>
#include <miopen/solver_id.hpp>
#include <nlohmann/json.hpp>
#include <set>
void mi(miopenStatus_t s) {
  require(s == miopenStatusSuccess, miopenGetErrorString(s));
}
struct Api {
  void *library;
  decltype(&ck_impl_fwd_fill_valid_kernels) fill;
  decltype(&ck_impl_fwd_is_applicable) app;
  decltype(&ck_impl_fwd_is_args_supported) args;
  decltype(&ck_impl_fwd_get_workspace_size) workspace;
  decltype(&ck_impl_fwd_get_solution) solution;
  decltype(&ck_impl_kernel_list_size) size;
  decltype(&ck_impl_kernel_list_get) get;
  decltype(&ck_impl_kernel_list_free) free_list;
  decltype(&ck_impl_solution_free) free_solution;
  explicit Api(const char *path) {
    library = dlopen(path, RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
    require(library != nullptr, "dlopen private plugin");
#define LOAD(field, symbol)                                                    \
  field = reinterpret_cast<decltype(field)>(dlsym(library, #symbol));          \
  require(field != nullptr, #symbol)
    LOAD(fill, ck_impl_fwd_fill_valid_kernels);
    LOAD(app, ck_impl_fwd_is_applicable);
    LOAD(args, ck_impl_fwd_is_args_supported);
    LOAD(workspace, ck_impl_fwd_get_workspace_size);
    LOAD(solution, ck_impl_fwd_get_solution);
    LOAD(size, ck_impl_kernel_list_size);
    LOAD(get, ck_impl_kernel_list_get);
    LOAD(free_list, ck_impl_kernel_list_free);
    LOAD(free_solution, ck_impl_solution_free);
#undef LOAD
  }
  std::vector<std::string> ids(const miopen::conv::ProblemDescription &p) {
    CKKernelListHandle *list = nullptr;
    size_t n = 0;
    require(fill(&p, miopenFloat, false, &list) == CK_IMPL_STATUS_SUCCESS &&
                list,
            "fill IDs");
    require(size(list, &n) == CK_IMPL_STATUS_SUCCESS, "list size");
    std::vector<std::string> result;
    for (size_t i = 0; i < n; ++i) {
      const char *s = nullptr;
      require(get(list, i, &s) == CK_IMPL_STATUS_SUCCESS && s, "list ID");
      result.emplace_back(s);
    }
    free_list(list);
    require(std::set<std::string>(result.begin(), result.end()).size() ==
                result.size(),
            "duplicate IDs");
    return result;
  }
};
struct Tensor {
  miopenTensorDescriptor_t d{};
  Tensor(int n, int c, int h, int w, bool nhwc) {
    mi(miopenCreateTensorDescriptor(&d));
    int lens[]{n, c, h, w};
    mi(miopenSetNdTensorDescriptorWithLayout(
        d, miopenFloat, nhwc ? miopenTensorNHWC : miopenTensorNCHW, lens, 4));
  }
  ~Tensor() { miopenDestroyTensorDescriptor(d); }
};

using Clock = std::chrono::steady_clock;
double ms(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}
struct Events {
  hipEvent_t begin, end;
  Events() {
    check(hipEventCreate(&begin));
    check(hipEventCreate(&end));
  }
  ~Events() {
    hipEventDestroy(begin);
    hipEventDestroy(end);
  }
  template <class F> nlohmann::json measure(F f, hipStream_t st, int repeats) {
    check(hipStreamSynchronize(st));
    auto t = Clock::now();
    check(hipEventRecord(begin, st));
    for (int i = 0; i < repeats; ++i)
      f();
    check(hipEventRecord(end, st));
    check(hipEventSynchronize(end));
    double wall = ms(t);
    float elapsed = 0;
    check(hipEventElapsedTime(&elapsed, begin, end));
    require(elapsed > 0 && std::isfinite(elapsed), "valid stream interval");
    return {{"repeats", repeats},
            {"stream_ms_per_call", elapsed / repeats},
            {"wall_ms_per_call", wall / repeats}};
  }
};
int main(int argc, char **argv) {
  try {
    require(argc >= 2, "private plugin path required");
    Api api(argv[1]);
    bool reverse = false, diagnostic = false;
    for (int i = 2; i < argc; ++i) {
      std::string arg = argv[i];
      require(arg == "--reverse" || arg == "--diagnostic", "unknown option");
      reverse |= arg == "--reverse";
      diagnostic |= arg == "--diagnostic";
    }
    hipDeviceProp_t props{};
    check(hipGetDeviceProperties(&props, 0));
    require(std::string(props.gcnArchName) == "gfx1013:xnack-",
            "native architecture");
    miopenHandle_t handle;
    mi(miopenCreate(&handle));
    Stream stream;
    mi(miopenSetStream(handle, stream.value));
    auto st = stream.value;
    auto &h = miopen::deref(handle);
    miopen::ExecutionContext context(&h);
    Events events;
    miopen::solver::Id direct("ConvHipDirectFwd"),
        public_dl("ConvHipImplicitGemmGroupFwdXdlops");
    require(direct.Value() == 218 && public_dl.Value() == 137,
            "fixed native solver IDs");
    nlohmann::json result = {{"status", "running"},
                             {"architecture", props.gcnArchName},
                             {"reverse", reverse},
                             {"diagnostic", diagnostic},
                             {"rows", nlohmann::json::array()}};
    size_t checkpoints = 0, calls = 0;
    std::set<std::string> observed;
    for (int size : {9, 32})
      for (int groups : {1, 3})
        for (int f : {1, 3}) {
          if (groups == 1 && f == 1)
            continue;
          int n = 2, c = size == 9 ? 4 : 16, k = size == 9 ? 8 : 16,
              height = size, width = size + 2, pad = f / 2;
          for (bool nhwc : {false, true}) {
            Tensor xd(n, groups * c, height, width, nhwc),
                wd(groups * k, c, f, f, nhwc),
                yd(n, groups * k, height, width, nhwc);
            miopenConvolutionDescriptor_t conv;
            mi(miopenCreateConvolutionDescriptor(&conv));
            mi(miopenInitConvolutionDescriptor(conv, miopenConvolution, pad,
                                               pad, 1, 1, 1, 1));
            mi(miopenSetConvolutionGroupCount(conv, groups));
            miopen::conv::ProblemDescription problem(
                miopen::deref(xd.d), miopen::deref(wd.d), miopen::deref(yd.d),
                miopen::deref(conv), miopen::conv::Direction::Forward);
            auto ids = api.ids(problem);
            require(ids.size() == (f == 1 ? 6 : 2), "expected admitted IDs");
            if (reverse)
              std::reverse(ids.begin(), ids.end());
            auto at = [=](int b, int ch, int y, int x, int channels, int rows,
                          int cols) {
              return nhwc ? ((b * rows + y) * cols + x) * channels + ch
                          : ((b * channels + ch) * rows + y) * cols + x;
            };
            // Identical logical values across both layouts, including filter
            // channel order.
            std::vector<float> a(n * groups * c * height * width),
                b(groups * k * c * f * f);
            for (int z = 0; z < n; ++z)
              for (int ch = 0; ch < groups * c; ++ch)
                for (int y = 0; y < height; ++y)
                  for (int x = 0; x < width; ++x) {
                    size_t logical =
                        ((z * groups * c + ch) * height + y) * width + x;
                    a[at(z, ch, y, x, groups * c, height, width)] =
                        float(int((logical * 13 + 3) % 19) - 9) / 11.f;
                  }
            for (int z = 0; z < groups * k; ++z)
              for (int ch = 0; ch < c; ++ch)
                for (int y = 0; y < f; ++y)
                  for (int x = 0; x < f; ++x) {
                    size_t logical = ((z * c + ch) * f + y) * f + x;
                    b[at(z, ch, y, x, c, f, f)] =
                        float(int((logical * 7 + 5) % 17) - 8) / 11.f;
                  }
            std::vector<double> gold(n * groups * k * height * width);
            for (int z = 0; z < n; ++z)
              for (int g = 0; g < groups; ++g)
                for (int out = 0; out < k; ++out)
                  for (int y = 0; y < height; ++y)
                    for (int x = 0; x < width; ++x) {
                      double sum = 0;
                      for (int ch = 0; ch < c; ++ch)
                        for (int fy = 0; fy < f; ++fy)
                          for (int fx = 0; fx < f; ++fx) {
                            int iy = y + fy - pad, ix = x + fx - pad;
                            if (iy >= 0 && iy < height && ix >= 0 && ix < width)
                              sum +=
                                  double(a[at(z, g * c + ch, iy, ix, groups * c,
                                              height, width)]) *
                                  double(
                                      b[at(g * k + out, ch, fy, fx, c, f, f)]);
                          }
                      gold[at(z, g * k + out, y, x, groups * k, height,
                              width)] = sum;
                    }
            Buffer<float> da(a, st), db(b, st), de(gold.size());
            size_t direct_space = 0, public_space = 0;
            std::optional<miopen::Invoker> direct_fn;
            if (!nhwc) {
              mi(miopenConvolutionForwardGetSolutionWorkspaceSize(
                  handle, wd.d, xd.d, conv, yd.d, direct.Value(),
                  &direct_space));
              mi(miopenConvolutionForwardCompileSolution(
                  handle, wd.d, xd.d, conv, yd.d, direct.Value()));
              direct_fn = h.GetInvoker(problem.MakeNetworkConfig(), direct);
              require(direct_fn.has_value(), "cached HIP-direct invoker");
              mi(miopenConvolutionForwardGetSolutionWorkspaceSize(
                  handle, wd.d, xd.d, conv, yd.d, public_dl.Value(),
                  &public_space));
              mi(miopenConvolutionForwardCompileSolution(
                  handle, wd.d, xd.d, conv, yd.d, public_dl.Value()));
            }
            for (const auto &id : ids) {
              observed.insert(id);
              auto started = Clock::now();
              miopen::solver::ConvSolution *solution = nullptr;
              require(api.solution(&context, &problem, id.c_str(), false,
                                   &solution) == CK_IMPL_STATUS_SUCCESS &&
                          solution && solution->invoker_factory,
                      "per-ID owning solution");
              size_t space = solution->workspace_sz;
              require(std::max({space, direct_space, public_space}) <
                          64 * 1024 * 1024,
                      "workspace cap");
              auto invoke = h.PrepareInvoker(*solution->invoker_factory,
                                             solution->construction_params);
              api.free_solution(solution);
              check(hipStreamSynchronize(st));
              double prepare_ms = ms(started);
              Buffer<unsigned char> work(
                  std::max({space, direct_space, public_space}));
              miopen::conv::DataInvokeParams data(
                  {miopen::deref(xd.d), da.data, miopen::deref(wd.d), db.data,
                   miopen::deref(yd.d), de.data},
                  space ? work.data : nullptr, space, false);
              auto direct_data = data;
              direct_data.workSpace = direct_space ? work.data : nullptr;
              direct_data.workSpaceSize = direct_space;
              auto verify = [&] {
                auto actual = de.read(gold.size(), st);
                double maximum = 0;
                for (size_t i = 0; i < gold.size(); ++i) {
                  double error = std::abs(double(actual[i]) - gold[i]);
                  require(std::isfinite(actual[i]) &&
                              error <= 2e-5 + 5e-5 * std::abs(gold[i]),
                          "CPU double reference");
                  maximum = std::max(maximum, error);
                }
                work.read(work.n, st);
                da.unchanged(a, st);
                db.unchanged(b, st);
                ++checkpoints;
                return maximum;
              };
              auto run = [&](int which) {
                if (which == 0)
                  invoke(h, data);
                else if (which == 1) {
                  require(direct_fn.has_value(), "HIP-direct only NCHW");
                  (*direct_fn)(h, direct_data);
                } else {
                  auto solver = which == 2 ? direct : public_dl;
                  size_t bytes = which == 2 ? direct_space : public_space;
                  mi(miopenConvolutionForwardImmediate(
                      handle, wd.d, db.data, xd.d, da.data, conv, yd.d, de.data,
                      bytes ? work.data : nullptr, bytes, solver.Value()));
                }
                ++calls;
              };
              nlohmann::json row = {{"h", height},
                                    {"w", width},
                                    {"groups", groups},
                                    {"filter", f},
                                    {"c_per_group", c},
                                    {"k_per_group", k},
                                    {"layout", nhwc ? "NHWC" : "NCHW"},
                                    {"kernel_id", id},
                                    {"workspace_bytes", space},
                                    {"prepare_ms", prepare_ms},
                                    {"windows", nlohmann::json::array()}};
              run(0);
              check(hipStreamSynchronize(st));
              row["first_max_abs_error"] = verify();
              if (!nhwc) {
                run(1);
                check(hipStreamSynchronize(st));
                verify();
              }
              if (!diagnostic) {
                int paths = nhwc ? 1 : 2;
                for (int j = 0; j < paths; ++j) {
                  for (int z = 0; z < 4; ++z)
                    run(j);
                  check(hipStreamSynchronize(st));
                  verify();
                }
                for (int cycle = 0; cycle < 3; ++cycle) {
                  std::vector<int> order = nhwc ? std::vector<int>{0, 0}
                                           : reverse
                                               ? std::vector<int>{1, 0, 0, 1}
                                               : std::vector<int>{0, 1, 1, 0};
                  for (int j : order) {
                    auto window = events.measure([&] { run(j); }, st, 32);
                    window["path"] = j;
                    window["cycle"] = cycle;
                    window["max_abs_error"] = verify();
                    row["windows"].push_back(window);
                  }
                }
                // One representative ID per layout anchors public-API versus
                // cached entry cost.
                if (!nhwc && id == ids.front())
                  for (int path : {2, 3})
                    for (int cycle = 0; cycle < 3; ++cycle) {
                      auto window = events.measure([&] { run(path); }, st, 32);
                      window["path"] = path;
                      window["cycle"] = cycle;
                      window["max_abs_error"] = verify();
                      row["public_windows"].push_back(window);
                    }
              }
              std::cerr << "B075_ROW " << row.dump() << "\n";
              result["rows"].push_back(row);
            }
            mi(miopenDestroyConvolutionDescriptor(conv));
          }
        }
    require(observed.size() == 6, "all six IDs observed");
    mi(miopenDestroy(handle));
    result["status"] = "passed";
    result["guards"] = true;
    result["checkpoints"] = checkpoints;
    result["convolution_calls"] = calls;
    result["kernel_ids"] = observed;
    std::cout << result.dump() << "\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
