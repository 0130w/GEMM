#pragma once

#include <cstdio>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t res = expr;                                                    \
    if (res != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Runtime Error: %s:%i:%d = %s\n", __FILE__,         \
              __LINE__, res, cudaGetErrorString(res));                         \
    }                                                                          \
  } while (0)

#define CUDA_CHECK_KERNEL()                                                    \
  do {                                                                         \
    CUDA_CHECK(cudaGetLastError());                                            \
    CUDA_CHECK(cudaDeviceSynchronize());                                       \
  } while (0)
