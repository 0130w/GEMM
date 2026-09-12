#include <cmath>
#include <cstdlib>
#include <cuda.h>

__global__ void naive_gemm(const float *__restrict__ A,
                           const float *__restrict__ B, float *__restrict__ C,
                           int M, int K, int N) {
  // 每个thread负责C中的一个元素
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) {
    return;
  }
  float acc = 0.f;
  for (int k = 0; k < K; ++k) {
    acc = fmaf(A[row * K + k], B[k * N + col], acc);
  }
  C[row * N + col] = acc;
}