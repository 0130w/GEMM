# GEMM

Some implemention of genernal matrix multiply.

## run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## todo

- [x] add cublas beat matching
- [ ] fix shared_mem_gemm