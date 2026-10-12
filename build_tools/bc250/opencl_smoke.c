// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(call)                                                            \
  do {                                                                         \
    cl_int code = (call);                                                      \
    if (code != CL_SUCCESS) {                                                  \
      fprintf(stderr, "%s: OpenCL error %d\n", #call, code);                   \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static const char *source =
    "__kernel void arithmetic(__global const float *x, __global float *y, "
    "__global float *sums, __global uint *count, uint n, __local float "
    "*scratch) {"
    "uint i=get_global_id(0), lane=get_local_id(0);"
    "float v=i<n ? 2*x[i]+3 : 0;"
    "if(i<n) {y[i]=v; atomic_inc(count);}"
    "scratch[lane]=v; barrier(CLK_LOCAL_MEM_FENCE);"
    "for(uint step=get_local_size(0)/2; step; step/=2) {"
    "if(lane<step) scratch[lane]+=scratch[lane+step];"
    "barrier(CLK_LOCAL_MEM_FENCE);}"
    "if(lane==0) sums[get_group_id(0)]=scratch[0];}"
    "__kernel void image_copy(read_only image2d_t image, __global uchar *out, "
    "uint width) { int x=get_global_id(0), y=get_global_id(1);"
    "uint4 p=read_imageui(image, CLK_NORMALIZED_COORDS_FALSE | "
    "CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(width-1-x,y));"
    "vstore4(convert_uchar4(p), y*width+x, out);}";

int main(void) {
  cl_uint platform_count = 0;
  CHECK(clGetPlatformIDs(0, NULL, &platform_count));
  if (platform_count != 1)
    return 2;
  cl_platform_id platform;
  cl_device_id device;
  cl_uint device_count = 0;
  CHECK(clGetPlatformIDs(1, &platform, NULL));
  CHECK(
      clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &device_count));
  if (device_count != 1)
    return 3;
  char name[512], version[512], extensions[8192];
  CHECK(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL));
  CHECK(clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(version), version,
                        NULL));
  CHECK(clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(extensions),
                        extensions, NULL));
  printf("OPENCL_DEVICE %s %s\n", name, version);
  cl_int code;
  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &code);
  CHECK(code);
  cl_command_queue queue =
      clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &code);
  CHECK(code);
  cl_program program =
      clCreateProgramWithSource(context, 1, &source, NULL, &code);
  CHECK(code);
  code = clBuildProgram(program, 1, &device, "-cl-std=CL1.2", NULL, NULL);
  if (code != CL_SUCCESS) {
    char log[32768];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log),
                          log, NULL);
    fprintf(stderr, "%s\n", log);
    return 4;
  }
  cl_kernel kernel = clCreateKernel(program, "arithmetic", &code);
  CHECK(code);
  const cl_uint n = 65539;
  const size_t local = 256, global = (n + local - 1) / local * local;
  const size_t blocks = global / local;
  float *x = malloc(n * sizeof(float)), *y = malloc(n * sizeof(float));
  float *sums = malloc(blocks * sizeof(float));
  if (!x || !y || !sums)
    return 5;
  for (cl_uint i = 0; i < n; ++i)
    x[i] = (float)((int)(i % 127) - 63);
  cl_mem input =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     n * sizeof(float), x, &code);
  CHECK(code);
  cl_mem output = clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(float),
                                 NULL, &code);
  CHECK(code);
  cl_mem partials = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   blocks * sizeof(float), NULL, &code);
  CHECK(code);
  cl_uint count = 0;
  cl_mem counter =
      clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                     sizeof(count), &count, &code);
  CHECK(code);
  CHECK(clSetKernelArg(kernel, 0, sizeof(input), &input));
  CHECK(clSetKernelArg(kernel, 1, sizeof(output), &output));
  CHECK(clSetKernelArg(kernel, 2, sizeof(partials), &partials));
  CHECK(clSetKernelArg(kernel, 3, sizeof(counter), &counter));
  CHECK(clSetKernelArg(kernel, 4, sizeof(n), &n));
  CHECK(clSetKernelArg(kernel, 5, local * sizeof(float), NULL));
  cl_event event;
  CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL,
                               &event));
  CHECK(clEnqueueReadBuffer(queue, output, CL_TRUE, 0, n * sizeof(float), y, 1,
                            &event, NULL));
  CHECK(clEnqueueReadBuffer(queue, partials, CL_TRUE, 0, blocks * sizeof(float),
                            sums, 0, NULL, NULL));
  CHECK(clEnqueueReadBuffer(queue, counter, CL_TRUE, 0, sizeof(count), &count,
                            0, NULL, NULL));
  float expected_sum = 0, actual_sum = 0;
  for (cl_uint i = 0; i < n; ++i) {
    float expected = 2 * x[i] + 3;
    if (y[i] != expected)
      return 6;
    expected_sum += expected;
  }
  for (size_t i = 0; i < blocks; ++i)
    actual_sum += sums[i];
  if (actual_sum != expected_sum || count != n)
    return 7;
  cl_ulong started, finished;
  CHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                sizeof(started), &started, NULL));
  CHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                sizeof(finished), &finished, NULL));
  if (!started || finished <= started)
    return 8;
  puts("OPENCL_ARITHMETIC_LDS_ATOMICS_EVENTS_PASS");

  cl_bool images = CL_FALSE;
  CHECK(clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof(images),
                        &images, NULL));
  if (images) {
    const size_t width = 17, height = 13, bytes = width * height * 4;
    unsigned char pixels[17 * 13 * 4], result[17 * 13 * 4];
    for (size_t i = 0; i < bytes; ++i)
      pixels[i] = (unsigned char)((i * 17 + 5) % 256);
    cl_image_format format = {CL_RGBA, CL_UNSIGNED_INT8};
    cl_image_desc descriptor = {0};
    descriptor.image_type = CL_MEM_OBJECT_IMAGE2D;
    descriptor.image_width = width;
    descriptor.image_height = height;
    cl_mem image =
        clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format,
                      &descriptor, pixels, &code);
    CHECK(code);
    cl_mem image_output =
        clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &code);
    CHECK(code);
    cl_kernel image_kernel = clCreateKernel(program, "image_copy", &code);
    CHECK(code);
    cl_uint w = width;
    CHECK(clSetKernelArg(image_kernel, 0, sizeof(image), &image));
    CHECK(clSetKernelArg(image_kernel, 1, sizeof(image_output), &image_output));
    CHECK(clSetKernelArg(image_kernel, 2, sizeof(w), &w));
    size_t size[2] = {width, height};
    CHECK(clEnqueueNDRangeKernel(queue, image_kernel, 2, NULL, size, NULL, 0,
                                 NULL, NULL));
    CHECK(clEnqueueReadBuffer(queue, image_output, CL_TRUE, 0, bytes, result, 0,
                              NULL, NULL));
    for (size_t y_idx = 0; y_idx < height; ++y_idx)
      for (size_t x_idx = 0; x_idx < width; ++x_idx)
        for (size_t channel = 0; channel < 4; ++channel)
          if (result[(y_idx * width + x_idx) * 4 + channel] !=
              pixels[(y_idx * width + width - 1 - x_idx) * 4 + channel])
            return 9;
    CHECK(clReleaseKernel(image_kernel));
    CHECK(clReleaseMemObject(image_output));
    CHECK(clReleaseMemObject(image));
    puts("OPENCL_IMAGE_SAMPLING_PASS");
  } else {
    puts("OPENCL_IMAGES_NOT_EXPOSED");
  }
  CHECK(clReleaseEvent(event));
  CHECK(clReleaseMemObject(counter));
  CHECK(clReleaseMemObject(partials));
  CHECK(clReleaseMemObject(output));
  CHECK(clReleaseMemObject(input));
  CHECK(clReleaseKernel(kernel));
  CHECK(clReleaseProgram(program));
  CHECK(clReleaseCommandQueue(queue));
  CHECK(clReleaseContext(context));
  free(sums);
  free(y);
  free(x);
  puts("OPENCL_NATIVE_GFX1013_PASS");
  return 0;
}
