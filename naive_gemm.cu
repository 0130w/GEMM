#include "common.h"
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

__host__ void init(float *__restrict__ A, float *__restrict__ B, int M, int K,
                   int N) {
  std::srand(42);
  for (int i = 0; i < M * K; ++i) {
    A[i] = (float)rand() / RAND_MAX;
  }
  for (int i = 0; i < K * N; ++i) {
    B[i] = (float)rand() / RAND_MAX;
  }
}

int main() {
  constexpr int M = 256, N = 256, K = 128;
  constexpr int blockDim_x = 16, blockDim_y = 16;
  constexpr int gridDim_x = (M + blockDim_x - 1) / blockDim_x;
  constexpr int gridDim_y = (N + blockDim_y - 1) / blockDim_y;
  dim3 block(blockDim_x, blockDim_y);
  dim3 grid(gridDim_x, gridDim_y);

  constexpr int bytes_A = M * K * sizeof(float);
  constexpr int bytes_B = K * N * sizeof(float);
  constexpr int bytes_C = M * N * sizeof(float);

  float *h_A, *h_B, *h_C;
  CUDA_CHECK(cudaMallocHost(&h_A, bytes_A));
  CUDA_CHECK(cudaMallocHost(&h_B, bytes_B));
  CUDA_CHECK(cudaMallocHost(&h_C, bytes_C));

  init(h_A, h_B, M, K, N);

  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
  CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
  CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

  CUDA_CHECK(cudaMemcpy(h_A, d_A, bytes_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(h_B, d_B, bytes_B, cudaMemcpyHostToDevice));

  naive_gemm<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
  CUDA_CHECK_KERNEL();

  CUDA_CHECK(cudaMemcpy(d_C, h_C, bytes_C, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFreeHost(h_A));
  CUDA_CHECK(cudaFreeHost(h_B));
  CUDA_CHECK(cudaFreeHost(h_C));
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
}