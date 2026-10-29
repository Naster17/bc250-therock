// SPDX-License-Identifier: MIT
// Explicit per-ID invokers, scalar admission and pre-dispatch runtime
// contracts.
#include "primitives_n216_common.hpp"
#include <array>
#include <dlfcn.h>
#include <iostream>
#include <map>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/conv_solution.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/handle.hpp>
#include <miopen/solver/ck_impl_interface.hpp>
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
struct Shape {
  int h, w, c, k, f, stride, dilation, pad;
};
const std::array<Shape, 5> shapes{{{5, 7, 4, 4, 1, 1, 1, 0},
                                   {9, 11, 4, 8, 1, 2, 1, 0},
                                   {9, 10, 4, 132, 3, 2, 1, 1},
                                   {8, 9, 8, 8, 3, 1, 2, 2},
                                   {7, 9, 3, 5, 3, 1, 1, 1}}};
const std::array<std::pair<double, double>, 8> bad_scalars{
    {{0, 0},
     {.5, 0},
     {std::nextafter(1., 2.), 0},
     {std::numeric_limits<double>::infinity(), 0},
     {std::numeric_limits<double>::quiet_NaN(), 0},
     {1, .25},
     {1, 1},
     {1, std::numeric_limits<double>::quiet_NaN()}}};
size_t api_refusals = 0, runtime_refusals = 0, executions = 0, fixtures = 0;
double max_error = 0;
std::map<std::string, size_t> counts;
std::set<std::pair<std::string, bool>> contracts;
void rejected_problem(Api &api, const miopen::ExecutionContext &context,
                      const miopen::conv::ProblemDescription &p) {
  bool value = true;
  require(api.app(&p, miopenFloat, false, &value) == CK_IMPL_STATUS_SUCCESS &&
              !value,
          "problem applicability refusal");
  require(api.ids(p).empty(), "problem empty kernel list");
  value = true;
  require(api.args(&p, "invalid", miopenFloat, false, &value) ==
                  CK_IMPL_STATUS_SUCCESS &&
              !value,
          "problem argument refusal");
  size_t bytes = 123;
  require(api.workspace(&p, miopenFloat, false, &bytes) ==
                  CK_IMPL_STATUS_INVALID_VALUE &&
              bytes == 123,
          "problem workspace refusal");
  miopen::solver::ConvSolution *out = nullptr;
  require(api.solution(&context, &p, "invalid", false, &out) ==
                  CK_IMPL_STATUS_INVALID_VALUE &&
              !out,
          "problem solution refusal");
  api_refusals += 5;
}
int main(int argc, char **argv) {
  try {
    require(argc == 2, "supply private DSO path");
    Api api(argv[1]);
    hipDeviceProp_t props{};
    check(hipGetDeviceProperties(&props, 0));
    require(std::string(props.gcnArchName) == "gfx1013:xnack-",
            "native architecture");
    miopenHandle_t handle;
    mi(miopenCreate(&handle));
    auto &h = miopen::deref(handle);
    miopen::ExecutionContext context(&h);
    Stream private_stream;
    for (bool nhwc : {false, true})
      for (int groups : {1, 3})
        for (size_t shape = 0; shape < shapes.size(); ++shape)
          for (bool custom : {false, true})
            for (bool dyadic : {false, true}) {
              const auto q = shapes[shape];
              const int n = 2;
              int oh =
                  (q.h + 2 * q.pad - q.dilation * (q.f - 1) - 1) / q.stride + 1;
              int ow =
                  (q.w + 2 * q.pad - q.dilation * (q.f - 1) - 1) / q.stride + 1;
              Tensor xd(n, groups * q.c, q.h, q.w, nhwc),
                  wd(groups * q.k, q.c, q.f, q.f, nhwc),
                  yd(n, groups * q.k, oh, ow, nhwc);
              miopenConvolutionDescriptor_t conv;
              mi(miopenCreateConvolutionDescriptor(&conv));
              mi(miopenInitConvolutionDescriptor(conv, miopenConvolution, q.pad,
                                                 q.pad, q.stride, q.stride,
                                                 q.dilation, q.dilation));
              mi(miopenSetConvolutionGroupCount(conv, groups));
              auto st = custom ? private_stream.value : hipStream_t{};
              mi(miopenSetStream(handle, st));
              miopen::conv::ProblemDescription p(
                  miopen::deref(xd.d), miopen::deref(wd.d), miopen::deref(yd.d),
                  miopen::deref(conv), miopen::conv::Direction::Forward);
              if (groups == 1 && shape == 0 && !custom && dyadic) {
                for (auto [alpha, beta] : bad_scalars) {
                  miopen::conv::ProblemDescription bad(
                      miopen::deref(xd.d), miopen::deref(wd.d),
                      miopen::deref(yd.d), miopen::deref(conv),
                      miopen::conv::Direction::Forward, 0,
                      miopen::Scalar(alpha), miopen::Scalar(beta));
                  rejected_problem(api, context, bad);
                }
                auto transposed = miopen::deref(conv);
                transposed.mode = miopenTranspose;
                miopen::conv::ProblemDescription transpose_problem(
                    miopen::deref(xd.d), miopen::deref(wd.d),
                    miopen::deref(yd.d), transposed,
                    miopen::conv::Direction::Forward);
                rejected_problem(api, context, transpose_problem);
                Tensor wrong_y(n, groups * q.k, oh, ow + 1, nhwc);
                miopen::conv::ProblemDescription wrong(
                    miopen::deref(xd.d), miopen::deref(wd.d),
                    miopen::deref(wrong_y.d), miopen::deref(conv),
                    miopen::conv::Direction::Forward);
                rejected_problem(api, context, wrong);
                Tensor huge_x(2, 4, 46341, 46341, nhwc),
                    huge_y(2, 4, 46341, 46341, nhwc);
                miopen::conv::ProblemDescription huge(
                    miopen::deref(huge_x.d), miopen::deref(wd.d),
                    miopen::deref(huge_y.d), miopen::deref(conv),
                    miopen::conv::Direction::Forward);
                rejected_problem(api, context, huge);
              }
              auto ids = api.ids(p);
              require(!ids.empty(), "no valid kernel for bounded fixture");
              auto at = [=](int b, int ch, int y, int x, int channels,
                            int height, int width) {
                return nhwc ? ((b * height + y) * width + x) * channels + ch
                            : ((b * channels + ch) * height + y) * width + x;
              };
              std::vector<float> a(n * groups * q.c * q.h * q.w),
                  b(groups * q.k * q.c * q.f * q.f);
              for (size_t i = 0; i < a.size(); ++i)
                a[i] =
                    float(int((i * 13 + 3) % 19) - 9) / (dyadic ? 8.f : 11.f);
              for (size_t i = 0; i < b.size(); ++i)
                b[i] = float(int((i * 7 + 5) % 17) - 8) / (dyadic ? 8.f : 11.f);
              std::vector<double> gold(n * groups * q.k * oh * ow);
              for (int batch = 0; batch < n; ++batch)
                for (int g = 0; g < groups; ++g)
                  for (int k = 0; k < q.k; ++k)
                    for (int y = 0; y < oh; ++y)
                      for (int x = 0; x < ow; ++x) {
                        double sum = 0;
                        for (int c = 0; c < q.c; ++c)
                          for (int fy = 0; fy < q.f; ++fy)
                            for (int fx = 0; fx < q.f; ++fx) {
                              int iy = y * q.stride + fy * q.dilation - q.pad,
                                  ix = x * q.stride + fx * q.dilation - q.pad;
                              if (iy >= 0 && iy < q.h && ix >= 0 && ix < q.w)
                                sum += double(a[at(batch, g * q.c + c, iy, ix,
                                                   groups * q.c, q.h, q.w)]) *
                                       double(b[at(g * q.k + k, c, fy, fx, q.c,
                                                   q.f, q.f)]);
                            }
                        gold[at(batch, g * q.k + k, y, x, groups * q.k, oh,
                                ow)] = sum;
                      }
              Buffer<float> da(a, st), db(b, st);
              for (const auto &id : ids) {
                bool supported = false;
                require(api.args(&p, id.c_str(), miopenFloat, false,
                                 &supported) == CK_IMPL_STATUS_SUCCESS &&
                            supported,
                        "listed ID supported");
                size_t workspace = 0;
                // Destroy the temporary problem and returned solution before
                // invocation.
                auto invoke = [&] {
                  auto temporary = p;
                  miopen::solver::ConvSolution *solution = nullptr;
                  require(api.solution(&context, &temporary, id.c_str(), false,
                                       &solution) == CK_IMPL_STATUS_SUCCESS &&
                              solution && solution->invoker_factory,
                          "specific-ID solution");
                  workspace = solution->workspace_sz;
                  require(workspace < 64 * 1024 * 1024, "workspace bound");
                  auto fn = h.PrepareInvoker(*solution->invoker_factory,
                                             solution->construction_params);
                  api.free_solution(solution);
                  return fn;
                }();
                Buffer<float> de(gold.size());
                Buffer<unsigned char> work(workspace);
                miopen::conv::DataInvokeParams data(
                    {miopen::deref(xd.d), da.data, miopen::deref(wd.d), db.data,
                     miopen::deref(yd.d), de.data},
                    workspace ? work.data : nullptr, workspace, false);
                if (contracts.insert({id, nhwc}).second) {
                  auto reject = [&](miopen::conv::DataInvokeParams bad) {
                    auto output = de.bytes(st), scratch = work.bytes(st);
                    bool refused = false;
                    try {
                      invoke(h, bad);
                    } catch (const miopen::Exception &e) {
                      require(e.status == miopenStatusBadParm,
                              "wrong runtime refusal status");
                      refused = true;
                    }
                    require(refused, "bad invocation accepted");
                    check(hipStreamSynchronize(st));
                    require(de.bytes(st) == output && work.bytes(st) == scratch,
                            "rejected invocation wrote output/workspace");
                    da.unchanged(a, st);
                    db.unchanged(b, st);
                    ++runtime_refusals;
                  };
                  for (auto [alpha, beta] : bad_scalars) {
                    auto bad = data;
                    bad.alpha = miopen::Scalar(alpha);
                    bad.beta = miopen::Scalar(beta);
                    reject(bad);
                  }
                  {
                    auto bad = data;
                    bad.tensors.in = nullptr;
                    reject(bad);
                  }
                  {
                    auto bad = data;
                    bad.tensors.w = nullptr;
                    reject(bad);
                  }
                  {
                    auto bad = data;
                    bad.tensors.out = nullptr;
                    reject(bad);
                  }
                  {
                    auto bad = data;
                    bad.tensors.in =
                        reinterpret_cast<const char *>(da.data) + 1;
                    reject(bad);
                  }
                  {
                    auto bad = data;
                    bad.tensors.out = da.data;
                    reject(bad);
                  }
                  {
                    auto bad = data;
                    bad.tensors.out = da.data + 1;
                    reject(bad);
                  }
                  {
                    auto bad = data;
                    bad.tensors.out = db.data;
                    reject(bad);
                  }
                  {
                    auto bad = data;
                    bad.tensors.outDesc = miopen::TensorDescriptor(
                        miopenFloat, {n, groups * q.k, oh, ow + 1});
                    reject(bad);
                  }
                  {
                    auto bad = data;
                    bad.tensors.inDesc = miopen::TensorDescriptor(
                        miopenFloat, {n, groups * q.c, q.h, q.w + 1});
                    reject(bad);
                  }
                  {
                    auto bad = data;
                    bad.tensors.wDesc = miopen::TensorDescriptor(
                        miopenFloat, {groups * q.k, q.c, q.f, q.f + 1});
                    reject(bad);
                  }
                  if (workspace) {
                    {
                      auto bad = data;
                      bad.workSpace = nullptr;
                      reject(bad);
                    }
                    {
                      auto bad = data;
                      bad.workSpaceSize = workspace - 1;
                      reject(bad);
                    }
                    {
                      auto bad = data;
                      bad.workSpace = da.data;
                      reject(bad);
                    }
                    {
                      auto bad = data;
                      bad.workSpace = de.data;
                      reject(bad);
                    }
                    {
                      auto bad = data;
                      bad.workSpace = work.data + 1;
                      reject(bad);
                    }
                  }
                  miopen::solver::ConvSolution *failed = nullptr;
                  require(api.solution(&context, &p, "not-a-kernel", false,
                                       &failed) ==
                                  CK_IMPL_STATUS_INVALID_VALUE &&
                              !failed,
                          "unknown ID refused without output change");
                  ++api_refusals;
                }
                invoke(h, data);
                auto result = de.read(gold.size(), st);
                work.read(workspace, st);
                for (size_t i = 0; i < gold.size(); ++i) {
                  double error = std::abs(double(result[i]) - gold[i]);
                  require(
                      std::isfinite(result[i]) &&
                          error <=
                              (dyadic ? 0. : 2e-5 + 5e-5 * std::abs(gold[i])),
                      "specific-ID CPU reference mismatch: " + id);
                  max_error = std::max(max_error, error);
                }
                da.unchanged(a, st);
                db.unchanged(b, st);
                ++executions;
                ++counts[id];
              }
              ++fixtures;
              mi(miopenDestroyConvolutionDescriptor(conv));
            }
    require(fixtures == 80 && counts.size() == 6,
            "bounded fixture/ID coverage");
    mi(miopenDestroy(handle));
    std::cout << "{\"status\":\"passed\",\"fixtures\":" << fixtures
              << ",\"executions\":" << executions
              << ",\"api_refusals\":" << api_refusals
              << ",\"runtime_refusals\":" << runtime_refusals
              << ",\"max_abs_error\":" << max_error
              << ",\"guards\":true,\"kernel_counts\":{";
    bool comma = false;
    for (auto [id, n] : counts) {
      if (comma)
        std::cout << ',';
      std::cout << '"' << id << "\":" << n;
      comma = true;
    }
    std::cout << "}}\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
