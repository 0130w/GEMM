# GEMM

Some implemention of genernal matrix multiply.

## run

```bash
make build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build -j
./build/gemm
```