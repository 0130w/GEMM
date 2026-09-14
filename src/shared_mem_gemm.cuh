#pragma once

#include "load_tile.cuh"
#include "store_tile.cuh"
#include <cmath>

template <int THREADNUM, int T_M, int T_N, int T_K>
__global__ void shared_mem_gemm(const float *__restrict__ A,
                                const float *__restrict__ B,
                                float *__restrict__ C, int M, int N, int K) {
  __shared__ float tileC[T_M * T_N];

  int tileRow = blockIdx.x * T_M;
  int tileCol = blockIdx.y * T_N;

  for (int i = 0; i < K; i += T_K) {
    __shared__ float tileA[T_M * T_K];
    __shared__ float tileB[T_K * T_N];
    // load A
    load_tile<THREADNUM, T_M, T_K>(tileA, A, tileRow, i, M, K);
    // load B
    load_tile<THREADNUM, T_K, T_N>(tileB, B, i, tileCol, K, N);
    __syncthreads();
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    for (int j = tid; j < T_M * T_N; j += THREADNUM) {
      // calculate tileC[r][c]
      int r = j / T_N;
      int c = j % T_N;
      float acc = 0.0f;
      for (int k = 0; k < T_K; ++k) {
        acc = fmaf(tileA[r * T_K + k], tileB[k * T_N + c], acc);
      }
      tileC[r * T_N + c] = acc;
    }
    store_tile<THREADNUM, T_M, T_N>(tileC, C, tileRow, tileCol, M, N);
    __syncthreads();
  }
  return;
}
