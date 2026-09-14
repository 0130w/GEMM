#pragma once

#include "common.h"
#include "cublas_v2.h"

inline void getCUBLASRes(int M, int N, int K, const float *__restrict__ A,
                         const float *__restrict__ B, float *C) {
  // A -> M * K; B -> K * N; C -> M * N
  float alpha = 1.f;
  float beta = 0.f;
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B,
                           N, A, K, &beta, C, N));
  CUBLAS_CHECK(cublasDestroy(handle));
}