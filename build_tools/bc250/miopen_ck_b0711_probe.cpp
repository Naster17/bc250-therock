// SPDX-License-Identifier: MIT
#include "primitives_n216_common.hpp"
#include <array>
#include <chrono>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <miopen/conv/problem_description.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/miopen.h>
#include <miopen/solver/ck_impl_interface.hpp>
#include <miopen/solver_id.hpp>
#include <nlohmann/json.hpp>
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

using Value = _Float16;
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}
int main(int argc, char **argv) {
  bool reverse = false, tune = false, diagnostic = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    require(arg == "--reverse" || arg == "--tune" || arg == "--diagnostic",
            "unknown argument");
    reverse |= arg == "--reverse";
    tune |= arg == "--tune";
    diagnostic |= arg == "--diagnostic";
  }

  try {
    hipDeviceProp_t props{};
    check(hipGetDeviceProperties(&props, 0));
    require(std::string(props.gcnArchName) == "gfx1013:xnack-",
            "native architecture");
    miopenHandle_t handle;
    mi(miopenCreate(&handle));
    Stream stream;
    mi(miopenSetStream(handle, stream.value));
    auto st = stream.value;
    miopen::solver::Id dl("ConvHipImplicitGemmGroupFwdXdlops"),
        direct("ConvHipDirectFwd");
    require(dl.IsValid() && direct.IsValid(), "solver IDs");
    nlohmann::json result = {{"status", "running"},
                             {"architecture", props.gcnArchName},
                             {"dl_solver", dl.Value()},
                             {"direct_solver", direct.Value()},
                             {"tune", tune},
                             {"diagnostic", diagnostic},
                             {"rows", nlohmann::json::array()}};
    size_t checkpoints = 0, calls = 0;
    // Six fixed shapes; same geometry as B07.5, NCHW public calls include
    // transforms.
    for (int size : {9, 32})
      for (int g : {1, 3})
        for (int f : {1, 3}) {
          if (g == 1 && f == 1)
            continue;
          const bool nhwc = false, dyadic = false;
          const int n = 2, c = size == 9 ? 4 : 16, k = size == 9 ? 8 : 16,
                    h = size, w = size + 2, pad = f / 2;
          Tensor xd(n, g * c, h, w, nhwc), wd(g * k, c, f, f, nhwc),
              yd(n, g * k, h, w, nhwc);
          miopenConvolutionDescriptor_t conv;
          mi(miopenCreateConvolutionDescriptor(&conv));
          mi(miopenInitConvolutionDescriptor(conv, miopenConvolution, pad, pad,
                                             1, 1, 1, 1));
          mi(miopenSetConvolutionGroupCount(conv, g));
          nlohmann::json row = {{"n", n},           {"c_per_group", c},
                                {"k_per_group", k}, {"h", h},
                                {"w", w},           {"groups", g},
                                {"filter", f},      {"layout", "NCHW"}};
          for (const char *key :
               {"workspace_query_ms", "compile_ms", "first_execute_wall_ms",
                "first_max_abs_error"})
            row[key] = nlohmann::json::array({0., 0.});
          std::array<size_t, 2> spaces{};
          size_t workspace = 0;
          std::array<uint64_t, 2> ids{dl.Value(), direct.Value()};
          for (int j : (reverse ? std::array<int, 2>{1, 0}
                                : std::array<int, 2>{0, 1})) {
            auto t = Clock::now();
            mi(miopenConvolutionForwardGetSolutionWorkspaceSize(
                handle, wd.d, xd.d, conv, yd.d, ids[j], &spaces[j]));
            row["workspace_query_ms"][j] = ms(t);
            require(spaces[j] < 64 * 1024 * 1024, "workspace cap");
            workspace = std::max(workspace, spaces[j]);
            if (!tune) {
              t = Clock::now();
              mi(miopenConvolutionForwardCompileSolution(handle, wd.d, xd.d,
                                                         conv, yd.d, ids[j]));
              check(hipStreamSynchronize(st));
              row["compile_ms"][j] = ms(t);
            }
          }
          auto at = [=](int b, int ch, int y, int x, int channels, int height,
                        int width) {
            return nhwc ? ((b * height + y) * width + x) * channels + ch
                        : ((b * channels + ch) * height + y) * width + x;
          };
          std::vector<Value> a(n * g * c * h * w), b(g * k * c * f * f);
          for (size_t i = 0; i < a.size(); ++i)
            a[i] = float(int((i * 13 + 3) % 19) - 9) / (dyadic ? 8.f : 11.f);
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
                            sum += double(a[at(batch, group * c + ch, iy, ix,
                                               g * c, h, w)]) *
                                   double(b[at(group * k + out, ch, fy, fx, c,
                                               f, f)]);
                        }
                    gold[at(batch, group * k + out, y, x, g * k, h, w)] = sum;
                  }
          Buffer<Value> da(a, st), db(b, st), de(gold.size());

          Buffer<unsigned char> work(workspace);
          auto verify = [&](int solver) {
            ++checkpoints;
            auto actual = de.read(gold.size(), st);
            double maxerr = 0;
            for (size_t i = 0; i < gold.size(); ++i) {
              double e = std::abs(double(actual[i]) - gold[i]);
              double spacing = std::max(
                  std::ldexp(1., -24),
                  std::ldexp(1., gold[i] == 0
                                     ? -24
                                     : std::ilogb(std::abs(gold[i])) - 10));
              const double bound =
                  .5 * spacing + 2e-5 + 5e-5 * std::abs(gold[i]);
              if (!std::isfinite(double(actual[i])) || e > bound) {
                std::cerr << "B0711_MISMATCH "
                          << nlohmann::json({{"solver_id", ids[solver]},
                                             {"h", h},
                                             {"w", w},
                                             {"groups", g},
                                             {"filter", f},
                                             {"index", i},
                                             {"actual", double(actual[i])},
                                             {"expected", gold[i]},
                                             {"error", e},
                                             {"bound", bound}})
                                 .dump()
                          << "\n";
                throw std::runtime_error("CPU-double reference mismatch");
              }
              maxerr = std::max(maxerr, e);
            }
            work.read(workspace, st);
            da.unchanged(a, st);
            db.unchanged(b, st);
            return maxerr;
          };
          auto invoke = [&](int j) {
            ++calls;
            mi(miopenConvolutionForwardImmediate(
                handle, wd.d, db.data, xd.d, da.data, conv, yd.d, de.data,
                spaces[j] ? work.data : nullptr, spaces[j], ids[j]));
          };

          if (tune) {
            auto started = Clock::now();
            int count = 0;
            miopenConvAlgoPerf_t perf{};
            mi(miopenFindConvolutionForwardAlgorithm(
                handle, xd.d, da.data, wd.d, db.data, conv, yd.d, de.data, 1,
                &count, &perf, workspace ? work.data : nullptr, workspace,
                true));
            check(hipStreamSynchronize(st));
            row["search_wall_ms"] = ms(started);
            require(count == 1 &&
                        perf.fwd_algo == miopenConvolutionFwdAlgoImplicitGEMM &&
                        perf.time > 0 && std::isfinite(perf.time),
                    "forced public search result");
            row["search_time_ms"] = perf.time;
            size_t found = 0;
            std::array<miopenConvSolution_t, 8> solutions{};
            mi(miopenConvolutionForwardGetSolution(handle, wd.d, xd.d, conv,
                                                   yd.d, solutions.size(),
                                                   &found, solutions.data()));
            require(found == 1 && solutions[0].solution_id == 137,
                    "genuine solver137 FindDb result");
            invoke(0);
            check(hipStreamSynchronize(st));
            row["max_abs_error"] = verify(0);
            row["windows"] = nlohmann::json::array();
            result["rows"].push_back(row);
            mi(miopenDestroyConvolutionDescriptor(conv));
            continue;
          }
          for (int j : (reverse ? std::array<int, 2>{1, 0}
                                : std::array<int, 2>{0, 1})) {
            auto t = Clock::now();
            invoke(j);
            check(hipStreamSynchronize(st));
            row["first_execute_wall_ms"][j] = ms(t);
            row["first_max_abs_error"][j] = verify(j);
          }
          if (diagnostic) {
            row["windows"] = nlohmann::json::array();
            result["rows"].push_back(row);
            mi(miopenDestroyConvolutionDescriptor(conv));
            continue;
          }
          hipEvent_t begin, end;
          check(hipEventCreate(&begin));
          check(hipEventCreate(&end));
          for (int j : (reverse ? std::array<int, 2>{1, 0}
                                : std::array<int, 2>{0, 1})) {
            for (int z = 0; z < 4; ++z)
              invoke(j);
            check(hipStreamSynchronize(st));
            verify(j);
          }
          constexpr int repeats = 32;
          for (int cycle = 0; cycle < 3; ++cycle)
            for (int j : (reverse ? std::array<int, 4>{1, 0, 0, 1}
                                  : std::array<int, 4>{0, 1, 1, 0})) {
              check(hipStreamSynchronize(st));
              auto t = Clock::now();
              check(hipEventRecord(begin, st));
              for (int z = 0; z < repeats; ++z)
                invoke(j);
              check(hipEventRecord(end, st));
              check(hipEventSynchronize(end));
              double wall = ms(t);
              float elapsed = 0;
              check(hipEventElapsedTime(&elapsed, begin, end));
              require(elapsed > 0 && std::isfinite(elapsed),
                      "invalid event interval");
              row["windows"].push_back(
                  {{"solver_index", j},
                   {"cycle", cycle},
                   {"repeats", repeats},
                   {"stream_interval_ms_per_call", elapsed / repeats},
                   {"wall_ms_per_call", wall / repeats},
                   {"max_abs_error", verify(j)}});
            }
          check(hipEventDestroy(begin));
          check(hipEventDestroy(end));
          mi(miopenDestroyConvolutionDescriptor(conv));
          result["rows"].push_back(row);
          std::cerr << "B0711_ROW " << row.dump() << "\n";
        }
    mi(miopenDestroy(handle));
    result["reference_checkpoints"] = checkpoints;
    result["explicit_immediate_calls"] = calls;
    result["dtype"] = "FP16-inputs-output-FP32-accumulator";
    result["status"] = "passed";
    result["guards"] = true;
    std::cout << result.dump() << "\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
