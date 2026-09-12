#include "common.h"
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
    load_tile<THREADNUM, T_M, T_K>(tileA, A, tileRow, i * T_K, M, K);
    // load B
    load_tile<THREADNUM, T_K, T_N>(tileB, B, i * T_K, tileCol, K, N);
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
  constexpr int T_M = 16, T_N = 16, T_K = 16;
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

  shared_mem_gemm<blockDim_x * blockDim_y, T_M, T_N, T_K>
      <<<grid, block>>>(d_A, d_B, d_C, M, K, N);
  CUDA_CHECK_KERNEL();

  CUDA_CHECK(cudaMemcpy(d_C, h_C, bytes_C, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFreeHost(h_A));
  CUDA_CHECK(cudaFreeHost(h_B));
  CUDA_CHECK(cudaFreeHost(h_C));
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
}