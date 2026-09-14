#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "naive_gemm.cuh"
#include "utils/common.h"
#include "utils/cublas_check.cuh"

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

  float *h_A, *h_B, *h_C, *h_C_ref;
  CUDA_CHECK(cudaMallocHost(&h_A, bytes_A));
  CUDA_CHECK(cudaMallocHost(&h_B, bytes_B));
  CUDA_CHECK(cudaMallocHost(&h_C, bytes_C));
  CUDA_CHECK(cudaMallocHost(&h_C_ref, bytes_C));

  init(h_A, h_B, M, K, N);

  float *d_A, *d_B, *d_C, *d_C_ref;
  CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
  CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
  CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
  CUDA_CHECK(cudaMalloc(&d_C_ref, bytes_C));

  CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

  // naive gemm
  naive_gemm<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
  CUDA_CHECK_KERNEL();
  // cublas
  getCUBLASRes(M, N, K, d_A, d_B, d_C_ref);

  CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_C_ref, d_C_ref, bytes_C, cudaMemcpyDeviceToHost));

  checkRes(h_C, h_C_ref, M * N);

  CUDA_CHECK(cudaFreeHost(h_A));
  CUDA_CHECK(cudaFreeHost(h_B));
  CUDA_CHECK(cudaFreeHost(h_C));
  CUDA_CHECK(cudaFreeHost(h_C_ref));
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  CUDA_CHECK(cudaFree(d_C_ref));
}
