#pragma once

#include "cublas_v2.h"
#include <cmath>
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

inline int checkRes(const float *__restrict__ got,
                    const float *__restrict__ ref, int N, float rtol = 1e-5f,
                    double atol = 1e-3) {
  for (int i = 0; i < N; ++i) {
    float g = got[i], r = ref[i];
    if (!std::isfinite(g)) {
      std::fprintf(stderr, "%s:%i: Check got[%d] = %lf is not finite\n",
                   __FILE__, __LINE__, i, g);
      return EXIT_FAILURE;
    }
    if (!std::isfinite(r)) {
      std::fprintf(stderr, "%s:%i: Check ref[%d] = %lf is not finite\n",
                   __FILE__, __LINE__, i, r);
      return EXIT_FAILURE;
    }
    double err = std::fabs(g - r);
    double tol = atol + rtol * std::fabs(r);
    if (err > tol) {
      std::fprintf(stderr, "%s:%i: result mismatch, err: %lf, tol: %lf\n",
                   __FILE__, __LINE__, err, tol);
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}