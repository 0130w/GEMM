#include "load_tile.cuh"
#include <__clang_cuda_builtin_vars.h>

template <int THREADNUM, int TR, int TC>
__device__ void load_tile(float *tile, const float *__restrict__ src, int r0,
                          int c0, int rows, int cols) {
  using ll = long long;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  constexpr int N = TR * TC;
  constexpr int STEP = (N + THREADNUM - 1) / THREADNUM;
  for (int k = 0; k < STEP; ++k) {
    int i = tid + k * THREADNUM;
    if (i < N) {
      int r = i / TC;
      int c = i % TC;
      int gr = r + r0;
      int gc = c + c0;
      tile[i] = (gr < rows && gc < cols) ? src[(ll)gr * cols + gc] : 0.0f;
    }
  }
}