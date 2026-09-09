__global__ void shared_mem_gemm(const float *__restrict__ A,
                                const float *__restrict__ B,
                                float *__restrict__ C, int M, int N, int K) {
  // A : M * K , B : K * N C : M * N
  // 从A中取B_M * B_K, B中取B_K * B_N 累加到对应的C B_M * B_N上
  // 取B_M = B_N = B_K = 16
  constexpr int B_M = 16, B_N = 16, B_K = 16;
  __shared__ float tileA[B_M * B_K]; // 256 * 4 bytes
  __shared__ float tileB[B_K * B_N];
  __shared__ float tileC[B_M * B_N];
  int rowA = blockIdx.x * B_M;
  int colA = blockIdx.y * B_K;
  

  return;
}

int main() {}