// SPDX-License-Identifier: MIT
// Real HIPRTC compilation and module execution, independent of MIGraphX linking.
#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

void checked(hipError_t status)
{
    if(status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
struct free_device { void operator()(void* p) const { (void)hipFree(p); } };
struct destroy_program { void operator()(hiprtcProgram* p) const { (void)hiprtcDestroyProgram(p); delete p; } };
struct unload_module { void operator()(void* p) const { (void)hipModuleUnload(static_cast<hipModule_t>(p)); } };

int main(int argc, char** argv)
{
    try
    {
        if(argc != 3) throw std::runtime_error("usage: probe SOURCE_ROOT baseline|patched");
        const bool baseline = std::string(argv[2]) == "baseline";
        hipDeviceProp_t props{};
        checked(hipGetDeviceProperties(&props, 0));
        if(std::string(props.gcnArchName).rfind("gfx1013:", 0) != 0 or props.warpSize != 32)
            throw std::runtime_error("Requires native gfx1013 wave32");
        const char* code = R"(
#include <migraphx/kernels/reduce.hpp>
extern "C" __global__ void jit_reduce(const float* input, float* output, unsigned n)
{
    auto idx = migraphx::make_index();
    float value = migraphx::block_reduce(idx, migraphx::op::sum{}, 0.0f, n,
                                         [&](auto j) { return input[j]; });
    if(idx.local == 0) output[0] = value;
}
)";
        int count = 0;
        for(int dpp : {1, 0})
        {
            auto program = std::unique_ptr<hiprtcProgram, destroy_program>(new hiprtcProgram{});
            if(hiprtcCreateProgram(program.get(), code, "b063.hip", 0, nullptr, nullptr) != HIPRTC_SUCCESS)
                throw std::runtime_error("hiprtcCreateProgram failed");
            std::vector<std::string> flags = {"--std=c++17", "--gpu-architecture=gfx1013", "-O2",
                "-DMIGRAPHX_USE_HIPRTC=1", "-DMIGRAPHX_NGLOBAL=128", "-DMIGRAPHX_NLOCAL=128",
                "-DMIGRAPHX_WAVEFRONTSIZE=32", "-Werror", "-Weverything",
                "-Wno-c++98-compat", "-Wno-c++98-compat-pedantic", "-Wno-conversion",
                "-Wno-double-promotion", "-Wno-exit-time-destructors", "-Wno-extra-semi",
                "-Wno-extra-semi-stmt", "-Wno-float-conversion", "-Wno-gnu-anonymous-struct",
                "-Wno-gnu-zero-variadic-macro-arguments", "-Wno-missing-prototypes",
                "-Wno-nested-anon-types", "-Wno-padded", "-Wno-shorten-64-to-32",
                "-Wno-sign-conversion", "-Wno-sign-compare", "-Wno-unused-command-line-argument",
                "-Wno-weak-vtables", "-Wno-c99-extensions", "-Wno-unsafe-buffer-usage",
                "-Wno-nrvo", "-Wno-lifetime-safety-intra-tu-suggestions",
                "-Wno-lifetime-safety-cross-tu-suggestions",
                std::string("-I") + argv[1] + "/src/targets/gpu/kernels/include",
                "-DMIGRAPHX_HAS_DPP=" + std::to_string(dpp)};
            std::vector<const char*> options;
            for(const auto& flag : flags) options.push_back(flag.c_str());
            auto status = hiprtcCompileProgram(*program, int(options.size()), options.data());
            size_t size = 0;
            hiprtcGetProgramLogSize(*program, &size);
            std::string log(size, '\0');
            hiprtcGetProgramLog(*program, log.data());
            std::cerr << "dpp=" << dpp << " status=" << int(status) << '\n' << log << '\n';
            if(baseline and dpp == 0)
            {
                if(status == HIPRTC_SUCCESS or log.find("C++20 extension") == std::string::npos)
                    throw std::runtime_error("Original strict C++17 failure not reproduced");
                ++count;
                continue;
            }
            if(status != HIPRTC_SUCCESS) throw std::runtime_error("Strict HIPRTC compile failed");
            ++count;
            if(baseline) continue;
            hiprtcGetCodeSize(*program, &size);
            std::vector<char> binary(size);
            if(hiprtcGetCode(*program, binary.data()) != HIPRTC_SUCCESS)
                throw std::runtime_error("hiprtcGetCode failed");
            hipModule_t raw_module{};
            checked(hipModuleLoadData(&raw_module, binary.data()));
            std::unique_ptr<void, unload_module> module{raw_module};
            hipFunction_t kernel{};
            checked(hipModuleGetFunction(&kernel, raw_module, "jit_reduce"));
            for(unsigned n : {33u, 129u, 513u})
            {
                std::vector<float> input(n + 2, 19), output(3, 29);
                for(unsigned i = 0; i < n; ++i) input[1 + i] = (int(i % 23) - 11) / 16.0f;
                auto original = input;
                float expected = float(std::accumulate(input.begin() + 1, input.end() - 1, 0.0));
                float *x{}, *y{};
                checked(hipMalloc(&x, input.size() * sizeof(float)));
                std::unique_ptr<float, free_device> dx{x};
                checked(hipMalloc(&y, output.size() * sizeof(float)));
                std::unique_ptr<float, free_device> dy{y};
                checked(hipMemcpy(x, input.data(), input.size() * sizeof(float), hipMemcpyHostToDevice));
                checked(hipMemcpy(y, output.data(), output.size() * sizeof(float), hipMemcpyHostToDevice));
                ++x; ++y;
                void* arguments[] = {&x, &y, &n};
                checked(hipModuleLaunchKernel(kernel, 1, 1, 1, 128, 1, 1, 0, nullptr, arguments, nullptr));
                checked(hipDeviceSynchronize());
                checked(hipMemcpy(output.data(), dy.get(), output.size() * sizeof(float), hipMemcpyDeviceToHost));
                checked(hipMemcpy(input.data(), dx.get(), input.size() * sizeof(float), hipMemcpyDeviceToHost));
                if(input != original or output != std::vector<float>{29, expected, 29})
                    throw std::runtime_error("HIPRTC module CPU-double/guard mismatch");
                ++count;
            }
        }
        std::cout << "{\"status\":\"passed\",\"checks\":" << count << "}\n";
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
