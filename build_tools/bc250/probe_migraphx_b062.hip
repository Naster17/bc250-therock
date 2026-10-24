// SPDX-License-Identifier: MIT
// Standalone admission probe using unchanged pinned MIGraphX kernel headers.
#include <migraphx/kernels/pointwise.hpp>
#include <migraphx/kernels/reduce.hpp>
#include <algorithm>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace migraphx {
template <> struct make_tensor<0>
{
    template <class T> __device__ static auto apply(T* p)
    {
        return make_tensor_view(p, make_shape(index_ints<3, 17>{}, index_ints<23, 1>{}));
    }
};
template <> struct make_tensor<1>
{
    template <class T> __device__ static auto apply(T* p)
    {
        return make_tensor_view(p, make_shape(index_ints<3, 17>{}, index_ints<1, 0>{}));
    }
};
template <> struct make_tensor<2>
{
    template <class T> __device__ static auto apply(T* p)
    {
        return make_tensor_view(p, make_shape(index_ints<3, 17>{}, index_ints<29, 1>{}));
    }
};
}

template <class T> MIGRAPHX_GLOBAL void pointwise_probe(T* x, T* bias, T* output)
{
    migraphx::pointwise<1, false>(migraphx::make_index())(
        [](auto a, auto b) {
            auto v = a * 0.5f + b;
            return migraphx::make_tuple(v > 0 ? v : 0);
        }, x, bias, output);
}

MIGRAPHX_GLOBAL void reduce_probe(const float* x, float* output, unsigned n)
{
    auto idx = migraphx::make_index();
    float value = migraphx::block_reduce(idx, migraphx::op::sum{}, 0.0f, n,
                                         [&](auto j) { return x[j]; });
    if(idx.local == 0) output[0] = value;
}

void check(hipError_t error)
{
    if(error != hipSuccess) throw std::runtime_error(hipGetErrorString(error));
}
struct gpu_free { void operator()(void* p) const { (void)hipFree(p); } };
template <class T> auto allocate(const std::vector<T>& values)
{
    T* p = nullptr;
    check(hipMalloc(&p, values.size() * sizeof(T)));
    std::unique_ptr<T, gpu_free> result{p};
    check(hipMemcpy(p, values.data(), values.size() * sizeof(T), hipMemcpyHostToDevice));
    return result;
}

template <class T> void pointwise_check()
{
    std::vector<T> x(3 * 23 + 4, T(19)), b(7, T(23)), y(3 * 29 + 4, T(29));
    auto expected = y;
    for(unsigned row = 0; row < 3; ++row)
    {
        b[1 + row] = T((int(row) - 1) / 4.0f);
        for(unsigned col = 0; col < 17; ++col)
        {
            x[1 + row * 23 + col] = T((int(row * 17 + col) - 25) / 16.0f);
            const double v = double(x[1 + row * 23 + col]) * 0.5 + double(b[1 + row]);
            expected[1 + row * 29 + col] = T(std::max(v, 0.0));
        }
    }
    auto dx = allocate(x), db = allocate(b), dy = allocate(y);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(pointwise_probe<T>), dim3(1), dim3(128), 0, 0,
                       dx.get() + 1, db.get() + 1, dy.get() + 1);
    check(hipGetLastError());
    check(hipDeviceSynchronize());
    check(hipMemcpy(y.data(), dy.get(), y.size() * sizeof(T), hipMemcpyDeviceToHost));
    if(y != expected) throw std::runtime_error("Pointwise CPU reference/output guard mismatch");
    auto saved_x = x, saved_b = b;
    check(hipMemcpy(x.data(), dx.get(), x.size() * sizeof(T), hipMemcpyDeviceToHost));
    check(hipMemcpy(b.data(), db.get(), b.size() * sizeof(T), hipMemcpyDeviceToHost));
    if(x != saved_x or b != saved_b) throw std::runtime_error("Input guard changed");
}

int main()
{
    try
    {
        hipDeviceProp_t props{};
        check(hipGetDeviceProperties(&props, 0));
        if(std::string(props.gcnArchName).rfind("gfx1013:", 0) != 0 or props.warpSize != 32)
            throw std::runtime_error("Probe requires native gfx1013 wave32");
        int passed = 1;
        pointwise_check<float>();
        ++passed;
        pointwise_check<migraphx::half>();
        ++passed;
        for(unsigned n : {1u, 17u, 31u, 32u, 33u, 63u, 64u, 127u, 128u, 129u, 257u, 513u})
        {
            std::vector<float> x(n + 4, 19), y(5, 29);
            for(unsigned i = 0; i < n; ++i) x[1 + i] = (int(i % 17) - 8) / 16.0f;
            const float expected = std::accumulate(x.begin() + 1, x.begin() + 1 + n, 0.0);
            auto dx = allocate(x), dy = allocate(y);
            hipLaunchKernelGGL(reduce_probe, dim3(1), dim3(128), 0, 0,
                               dx.get() + 1, dy.get() + 1, n);
            check(hipGetLastError());
            check(hipDeviceSynchronize());
            check(hipMemcpy(y.data(), dy.get(), y.size() * sizeof(float), hipMemcpyDeviceToHost));
            if(y != std::vector<float>{29, expected, 29, 29, 29})
                throw std::runtime_error("Reduction CPU-double reference/output guard mismatch");
            auto saved = x;
            check(hipMemcpy(x.data(), dx.get(), x.size() * sizeof(float), hipMemcpyDeviceToHost));
            if(x != saved) throw std::runtime_error("Reduction input guard changed");
            ++passed;
        }
        std::cout << "{\"status\":\"passed\",\"checks\":" << passed
                  << ",\"arch\":\"" << props.gcnArchName
                  << "\",\"wavefront\":" << props.warpSize
                  << ",\"dpp\":" << MIGRAPHX_HAS_DPP << "}\n";
    }
    catch(const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
