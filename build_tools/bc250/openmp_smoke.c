// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  const int n = 65539;
  float *x = malloc((size_t)n * sizeof(float));
  float *y = malloc((size_t)n * sizeof(float));
  if (!x || !y || omp_get_num_devices() < 1)
    return 1;
  for (int i = 0; i < n; ++i) {
    x[i] = (float)(i % 127 - 63);
    y[i] = (float)(i % 31);
  }
  int on_host = 1;
#pragma omp target map(from : on_host)
  { on_host = omp_is_initial_device(); }
  if (on_host)
    return 2;

  long long checksum = 0;
#pragma omp target teams distribute parallel for map(to : x[0 : n])            \
    map(tofrom : y[0 : n]) reduction(+ : checksum)
  for (int i = 0; i < n; ++i) {
    y[i] = 2 * x[i] + y[i];
    checksum += i % 127;
  }
  long long expected_checksum = 0;
  for (int i = 0; i < n; ++i) {
    if (y[i] != 2 * x[i] + (float)(i % 31))
      return 3;
    expected_checksum += i % 127;
  }
  if (checksum != expected_checksum)
    return 4;

#pragma omp target data map(to : x[0 : n]) map(from : y[0 : n])
  {
#pragma omp target teams distribute parallel for nowait depend(out : y[0 : n])
    for (int i = 0; i < n; ++i)
      y[i] = 3 * x[i];
#pragma omp taskwait
  }
  for (int i = 0; i < n; ++i)
    if (y[i] != 3 * x[i])
      return 5;
  free(x);
  free(y);
  puts("OPENMP_NATIVE_GFX1013_PASS");
  return 0;
}
