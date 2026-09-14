#pragma once

#include "cublas_v2.h"
#include <cstdio>
#include <cstdlib>
#include <stdlib.h>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t res = (expr);                                                  \
    if (res != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA Runtime Error: %s:%i:%d = %s\n", __FILE__,    \
                   __LINE__, res, cudaGetErrorString(res));                    \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#define CUDA_CHECK_KERNEL()                                                    \
  do {                                                                         \
    CUDA_CHECK(cudaGetLastError());                                            \
    CUDA_CHECK(cudaDeviceSynchronize());                                       \
  } while (0)

#define CUBLAS_CHECK(expr)                                                     \
  do {                                                                         \
    cublasStatus_t res = (expr);                                               \
    if (res != CUBLAS_STATUS_SUCCESS) {                                        \
      std::fprintf(stderr, "CUDA Runtime Error: %s:%i = %d\n", __FILE__,       \
                   __LINE__, (int)res);                                        \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

inline void init(float *__restrict__ A, float *__restrict__ B, int M, int K,
                 int N) {
  std::srand(42);
  for (int i = 0; i < M * K; ++i) {
    A[i] = (float)rand() / RAND_MAX;
  }
  for (int i = 0; i < K * N; ++i) {
    B[i] = (float)rand() / RAND_MAX;
  }
}
