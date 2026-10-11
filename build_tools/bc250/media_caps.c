// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Query kernel-visible engines without submitting GPU work.
#include <amdgpu.h>
#include <amdgpu_drm.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: media_caps <DRM render node>\n");
    return 1;
  }
  const int fd = open(argv[1], O_RDWR | O_CLOEXEC);
  if (fd < 0) {
    perror("open render node");
    return 1;
  }
  uint32_t major = 0, minor = 0;
  amdgpu_device_handle device = NULL;
  const int initialized = amdgpu_device_initialize(fd, &major, &minor, &device);
  if (initialized != 0) {
    fprintf(stderr, "amdgpu_device_initialize: %d\n", initialized);
    close(fd);
    return 1;
  }
  const struct {
    const char *name;
    unsigned type;
  } engines[] = {
      {"gfx", AMDGPU_HW_IP_GFX},
      {"compute", AMDGPU_HW_IP_COMPUTE},
      {"sdma", AMDGPU_HW_IP_DMA},
      {"uvd", AMDGPU_HW_IP_UVD},
      {"vce", AMDGPU_HW_IP_VCE},
      {"vcn_decode", AMDGPU_HW_IP_VCN_DEC},
      {"vcn_encode", AMDGPU_HW_IP_VCN_ENC},
      {"jpeg", AMDGPU_HW_IP_VCN_JPEG},
  };
  printf("{\"drm_version\":\"%u.%u\",\"engines\":[", major, minor);
  for (unsigned i = 0; i < sizeof(engines) / sizeof(engines[0]); ++i) {
    uint32_t count = 0;
    struct drm_amdgpu_info_hw_ip info = {0};
    const int count_status =
        amdgpu_query_hw_ip_count(device, engines[i].type, &count);
    const int info_status =
        amdgpu_query_hw_ip_info(device, engines[i].type, 0, &info);
    printf("%s{\"name\":\"%s\",\"count_status\":%d,\"count\":%u,"
           "\"info_status\":%d,\"version\":\"%u.%u\",\"rings\":%u}",
           i ? "," : "", engines[i].name, count_status, count, info_status,
           info.hw_ip_version_major, info.hw_ip_version_minor,
           info.available_rings);
  }
  puts("]}");
  amdgpu_device_deinitialize(device);
  close(fd);
  return 0;
}
