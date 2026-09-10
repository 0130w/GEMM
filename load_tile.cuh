#pragma once

template <int THREADNUM, int TR, int TC>
__device__ void load_tile(float *tile, const float *__restrict__ src, int r0,
                          int c0, int rows, int cols);