#pragma once

#include "load_tile.cuh"
#include <cmath>

template <int THREADNUM, int T_M, int T_N, int T_K>
__global__ void shared_mem_gemm(const float *__restrict__ A,
                                const float *__restrict__ B,
                                float *__restrict__ C, int M, int N, int K) {
  __shared__ float tileA[T_M * T_K];
  __shared__ float tileB[T_K * T_N];
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int tileRow = blockIdx.x * T_M;
  int tileCol = blockIdx.y * T_N;
  static_assert((T_M * T_N) % THREADNUM == 0,
                "THREADNUM must divide T_M * T_N.");
  constexpr int ELEMS = (T_M * T_N) / THREADNUM;
  float acc[ELEMS] = {0.f};

  for (int i = 0; i < K; i += T_K) {
    __syncthreads();
    load_tile<THREADNUM, T_M, T_K>(tileA, A, tileRow, i, M, K);
    load_tile<THREADNUM, T_K, T_N>(tileB, B, i, tileCol, K, N);
    __syncthreads();
#pragma unroll
    for (int e = 0; e < ELEMS; ++e) {
      int j = tid + e * THREADNUM;
      int r = j / T_N;
      int c = j % T_N;
      for (int k = 0; k < T_K; ++k) {
        acc[e] = fmaf(tileA[r * T_K + k], tileB[k * T_N + c], acc[e]);
      }
    }
  }

#pragma unroll
  for (int e = 0; e < ELEMS; ++e) {
    int j = tid + e * THREADNUM;
    int gr = tileRow + j / T_N, gc = tileCol + j % T_N;
    if (gr < M && gc < N) {
      C[gr * N + gc] = acc[e];
    }
  }
  return;
}
