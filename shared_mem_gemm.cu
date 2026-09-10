#include "load_tile.cuh"

template <int T_M, int T_N, int T_K>
__global__ void __launch_bounds__(T_M *T_N)
    shared_mem_gemm(const float *__restrict__ A, const float *__restrict__ B,
                    float *__restrict__ C, int M, int N, int K) {
  __shared__ float tileC[T_M * T_N];

  int tileRow = blockIdx.x * T_M;
  int tileCol = blockIdx.y * T_N;

  for (int i = 0; i < K; i += T_K) {
    // load A
    __shared__ float tileA[T_M * T_K];
    __shared__ float tileB[T_K * T_N];
    __syncthreads();
  }

  return;
}

int main() {}