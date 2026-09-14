#pragma once

#include "common.h"
#include "cublas_v2.h"

inline bool checkMatrixResByCublas(int M, int N, int K,
                                   const float *__restrict__ A,
                                   const float *__restrict__ B,
                                   const float *__restrict__ C) {
  // A -> M * K; B -> K * N; C -> M * N
  float alpha = 1.f;
  float beta = 0.f;
  bool res = false;
  float *dC;
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  CUDA_CHECK(cudaMalloc(&dC, sizeof(float) * M * N));
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, M, K, &alpha, B,
                           N, A, K, &beta, dC, N));
  // check dC and C
  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaFree(dC));
}